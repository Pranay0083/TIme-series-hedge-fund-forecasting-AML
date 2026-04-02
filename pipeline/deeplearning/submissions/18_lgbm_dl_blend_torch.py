#!/usr/bin/env python3
"""
Experiment 18 — Per-horizon blend: LightGBM (per sub_category) + DL seed ensemble.

For each horizon, trains LGBM models per sub_category (same recipe as
pipeline/lgbm/submissions/13) and a multi-seed PyTorch stack (same recipe as
experiment 17). Merges validation predictions by row id, grid-searches blend
weight alpha so pred = alpha * lgbm + (1 - alpha) * dl maximizes weighted RMSE
skill, then applies the same alpha on test rows for that horizon.

Outputs
    outputs/lgbm_dl_blend_torch_results.txt
    outputs/lgbm_dl_blend_torch_submission.csv

Runtime
    Roughly LGBM(13) + DL(17) for one full pass; reduce LGBM_SEEDS / DL_SEEDS for
    dry runs.
"""
import importlib.util
import inspect
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gc
import lightgbm as lgb
import numpy as np
import pandas as pd

_SUB_DIR = os.path.dirname(os.path.abspath(__file__))
_DL_ROOT = os.path.dirname(_SUB_DIR)
_LGBM_ROOT = os.path.join(os.path.dirname(_DL_ROOT), "lgbm")

for _p in (_DL_ROOT, _LGBM_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)


def import_dl_module(module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_DL_ROOT, f"{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_lgbm_module(module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_LGBM_ROOT, f"{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod_dl_paths = import_dl_module("01_paths")
mod_dl_log = import_dl_module("02_logging")
mod_dl_03 = import_dl_module("03_feature_engineering")
mod_dl_07 = import_dl_module("07_post_processing_ensembling")
mod_dl_08 = import_dl_module("08_pipeline_infrastructure")
mod_dl_09 = import_dl_module("09_torch_estimator")

mod_lgb_03 = import_lgbm_module("03_metrics")
mod_lgb_04 = import_lgbm_module("04_encoding_stats")
mod_lgb_06 = import_lgbm_module("06_features_enhanced")

TRAIN_PATH = mod_dl_paths.TRAIN_PATH
TEST_PATH = mod_dl_paths.TEST_PATH
OUTPUTS_DIR = mod_dl_paths.OUTPUTS_DIR
HORIZONS = mod_dl_paths.HORIZONS
VAL_THRESHOLD = mod_dl_paths.VAL_THRESHOLD
DL_MODELS_DIR = mod_dl_paths.DL_MODELS_DIR

setup_logging = mod_dl_log.setup_logging
create_spread_ratios = mod_dl_03.create_spread_ratios
create_group_z_scores = mod_dl_03.create_group_z_scores
create_rolling_lag_features = mod_dl_03.create_rolling_lag_features
ExpandingMeanTargetEncoder = mod_dl_03.ExpandingMeanTargetEncoder
apply_target_clipping = mod_dl_07.apply_target_clipping
aggressive_downcasting = mod_dl_08.aggressive_downcasting
MemoryMonitor = mod_dl_08.MemoryMonitor
HorizonTorchEstimator = mod_dl_09.HorizonTorchEstimator


def _horizon_torch_kw(torch_kw: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys the installed HorizonTorchEstimator does not accept (older 09_torch_estimator)."""
    params = inspect.signature(HorizonTorchEstimator.__init__).parameters
    allowed = {name for name in params if name != "self"}
    return {k: v for k, v in torch_kw.items() if k in allowed}


weighted_rmse_score = mod_lgb_03.weighted_rmse_score
compute_train_stats = mod_lgb_04.compute_train_stats
build_context_features = mod_lgb_06.build_context_features

DL_EXCLUDE_COLS = {
    "id",
    "code",
    "sub_code",
    "sub_category",
    "horizon",
    "ts_index",
    "weight",
    "y_target",
}

HORIZON_OVERRIDES = {
    1: {"latent_dim": 24, "drop_rate": 0.35, "patience": 30},
    3: {"latent_dim": 28, "drop_rate": 0.32, "patience": 28},
    10: {"latent_dim": 40, "drop_rate": 0.25, "patience": 35, "max_epochs": 250},
    25: {"latent_dim": 48, "drop_rate": 0.22, "patience": 40, "max_epochs": 280},
}

LGBM_CFG = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "n_estimators": 6000,
    "num_leaves": 96,
    "min_child_samples": 150,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.2,
    "lambda_l2": 15.0,
    "verbosity": -1,
}

LGBM_SEEDS = [42, 2024, 7, 11, 999]
DL_SEEDS = [42, 1042, 2042, 3042]


def build_advanced_features(df, target_encoder=None, is_train=True):
    with MemoryMonitor("Feature Engineering - Spreads & Ratios"):
        df = create_spread_ratios(df)
    with MemoryMonitor("Feature Engineering - Group Z-Scores"):
        df = create_group_z_scores(df)
    with MemoryMonitor("Feature Engineering - Rolling Lags"):
        df = create_rolling_lag_features(df)
    with MemoryMonitor("Feature Engineering - Target Encoding"):
        if "sub_category" in df.columns:
            if is_train and "y_target" in df.columns:
                target_encoder = ExpandingMeanTargetEncoder(cat_col="sub_category")
                df = target_encoder.fit_transform_sequential(df, "y_target")
            elif target_encoder is not None:
                df = target_encoder.transform(df)
    return df, target_encoder


def grid_search_alpha(
    y_true: np.ndarray,
    p_lgbm: np.ndarray,
    p_dl: np.ndarray,
    sw: np.ndarray,
    n_steps: int = 101,
) -> Tuple[float, float]:
    best_s = -1.0
    best_a = 0.5
    for i in range(n_steps):
        a = i / float(max(1, n_steps - 1))
        pred = a * p_lgbm + (1.0 - a) * p_dl
        s = weighted_rmse_score(y_true, pred, sw)
        if s > best_s:
            best_s = s
            best_a = a
    return best_a, best_s


def train_lgbm_horizon(
    hz: int,
    train_stats,
    sub_categories: List,
    logger: Any,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Val predictions (id, pred_lgbm) and test predictions (id, pred_lgbm) for this horizon."""
    logger.info(f"  LGBM: loading horizon {hz}...")
    hz_train = pd.read_parquet(TRAIN_PATH)
    hz_train = hz_train[hz_train["horizon"] == hz].copy()
    hz_train = build_context_features(hz_train, train_stats, hz, logger)

    has_test = TEST_PATH.exists()
    hz_test: Optional[pd.DataFrame] = None
    if has_test:
        hz_test = pd.read_parquet(TEST_PATH)
        hz_test = hz_test[hz_test["horizon"] == hz].copy()
        hz_test = build_context_features(hz_test, train_stats, hz, logger)

    exclude_cols = {
        "id",
        "code",
        "sub_code",
        "sub_category",
        "horizon",
        "ts_index",
        "weight",
        "y_target",
    }
    feature_cols = [c for c in hz_train.columns if c not in exclude_cols]

    val_rows: List[pd.DataFrame] = []
    tst_parts: List[pd.DataFrame] = []

    for sc in sub_categories:
        sc_train = hz_train[hz_train["sub_category"] == sc].copy()
        if len(sc_train) == 0:
            continue

        fit_mask = sc_train["ts_index"] <= VAL_THRESHOLD
        val_mask = sc_train["ts_index"] > VAL_THRESHOLD

        X_fit = sc_train.loc[fit_mask, feature_cols]
        y_fit = sc_train.loc[fit_mask, "y_target"]
        w_fit = sc_train.loc[fit_mask, "weight"]

        X_hold = sc_train.loc[val_mask, feature_cols]
        y_hold = sc_train.loc[val_mask, "y_target"]
        w_hold = sc_train.loc[val_mask, "weight"]

        if len(X_fit) == 0 or len(X_hold) == 0:
            del sc_train
            continue

        val_pred = np.zeros(len(y_hold))
        sc_test = None
        if has_test and hz_test is not None:
            sc_test = hz_test[hz_test["sub_category"] == sc]
        tst_pred = (
            np.zeros(len(sc_test)) if sc_test is not None and len(sc_test) > 0 else None
        )

        for seed in LGBM_SEEDS:
            mdl = lgb.LGBMRegressor(**LGBM_CFG, random_state=seed)
            mdl.fit(
                X_fit,
                y_fit,
                sample_weight=w_fit,
                eval_set=[(X_hold, y_hold)],
                eval_sample_weight=[w_hold],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            val_pred += mdl.predict(X_hold) / len(LGBM_SEEDS)
            if tst_pred is not None and sc_test is not None and len(sc_test) > 0:
                tst_pred += mdl.predict(sc_test[feature_cols]) / len(LGBM_SEEDS)

        ids_val = sc_train.loc[val_mask, "id"].values
        val_rows.append(pd.DataFrame({"id": ids_val, "pred_lgbm": val_pred}))

        if tst_pred is not None and sc_test is not None and len(sc_test) > 0:
            tst_parts.append(
                pd.DataFrame({"id": sc_test["id"].values, "pred_lgbm": tst_pred})
            )

        del sc_train
        gc.collect()

    del hz_train
    gc.collect()

    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    tst_df = pd.concat(tst_parts, ignore_index=True) if tst_parts else None
    return val_df, tst_df


def main():
    logger = setup_logging("LgbmDlBlend", "18_lgbm_dl_blend_torch.log")

    logger.info("=" * 70)
    logger.info("EXPERIMENT 18: LGBM + DL PER-HORIZON BLEND")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"LGBM seeds: {LGBM_SEEDS} | DL seeds: {DL_SEEDS}")

    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)
    _meta = pd.read_parquet(TRAIN_PATH, columns=["sub_category"])
    sub_categories: List = sorted(_meta["sub_category"].unique().tolist())
    del _meta
    gc.collect()

    train_raw = pd.read_parquet(TRAIN_PATH)
    train_raw = aggressive_downcasting(train_raw)
    train_fe, enc = build_advanced_features(train_raw, is_train=True)
    del train_raw
    gc.collect()

    test_fe = None
    has_test = TEST_PATH.exists()
    if has_test:
        test_raw = pd.read_parquet(TEST_PATH)
        test_raw = aggressive_downcasting(test_raw)
        test_fe, _ = build_advanced_features(test_raw, target_encoder=enc, is_train=False)
        del test_raw
        gc.collect()

    base_torch_kw = dict(
        latent_dim=32,
        drop_rate=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=2048,
        max_epochs=200,
        patience=25,
        val_frac_timesteps=0.12,
        loss="huber",
        huber_delta=1.0,
        lr_scheduler="cosine",
        cosine_min_lr=1e-5,
    )

    cv_y: List[float] = []
    cv_p: List[float] = []
    cv_w: List[float] = []
    cv_dl_only: List[float] = []
    cv_lgbm_only: List[float] = []
    horizon_alphas: Dict[int, float] = {}
    horizon_blend_scores: Dict[int, float] = {}
    horizon_dl_scores: Dict[int, float] = {}
    horizon_lgbm_scores: Dict[int, float] = {}
    test_outputs: List[pd.DataFrame] = []
    torch_kw_warned = False

    for hz in HORIZONS:
        logger.info("\n" + "=" * 70)
        logger.info(f"HORIZON = {hz}")
        logger.info("=" * 70)

        val_lgbm, tst_lgbm = train_lgbm_horizon(hz, train_stats, sub_categories, logger)
        if val_lgbm.empty:
            logger.warning(f"No LGBM val predictions for hz={hz}; skipping horizon.")
            continue

        tr_hz = train_fe[train_fe["horizon"] == hz].copy()
        te_hz = test_fe[test_fe["horizon"] == hz].copy() if has_test and test_fe is not None else None

        fit_mask = tr_hz.ts_index <= VAL_THRESHOLD
        val_mask = tr_hz.ts_index > VAL_THRESHOLD
        tr_fit = tr_hz.loc[fit_mask]
        tr_val = tr_hz.loc[val_mask]

        feature_cols = [
            c
            for c in tr_hz.columns
            if c not in DL_EXCLUDE_COLS and pd.api.types.is_numeric_dtype(tr_hz[c])
        ]

        y_hold = tr_val["y_target"].values
        w_hold = tr_val["weight"].values
        val_pred_dl = np.zeros(len(tr_val))
        tst_pred_dl = None
        if has_test and te_hz is not None and len(te_hz):
            tst_pred_dl = np.zeros(len(te_hz))

        torch_kw = {**base_torch_kw, **HORIZON_OVERRIDES.get(hz, {})}
        torch_kw_fit = _horizon_torch_kw(torch_kw)
        if not torch_kw_warned and set(torch_kw) - set(torch_kw_fit):
            logger.warning(
                "HorizonTorchEstimator missing knobs %s — using MSE + no cosine. "
                "Sync pipeline/deeplearning/09_torch_estimator.py from repo for huber/cosine.",
                sorted(set(torch_kw) - set(torch_kw_fit)),
            )
            torch_kw_warned = True

        for seed in DL_SEEDS:
            logger.info(f"  DL training seed {seed}...")
            est = HorizonTorchEstimator(horizon=hz, random_seed=seed, **torch_kw_fit)
            est.fit(
                tr_fit,
                tr_fit["y_target"].values,
                tr_fit["weight"].values if "weight" in tr_fit.columns else None,
                X_val=tr_val,
                y_val=tr_val["y_target"].values,
                w_val=tr_val["weight"].values if "weight" in tr_val.columns else None,
            )
            val_pred_dl += est.predict(tr_val) / len(DL_SEEDS)
            if tst_pred_dl is not None:
                raw = est.predict(te_hz)
                clipped = apply_target_clipping(raw, tr_fit["y_target"].values)
                tst_pred_dl += clipped / len(DL_SEEDS)
            if seed == DL_SEEDS[-1]:
                DL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
                est.save(str(DL_MODELS_DIR / f"18_blend_dl_horizon_{hz}_lastseed.pt"))

        val_dl = pd.DataFrame(
            {"id": tr_val["id"].values, "pred_dl": val_pred_dl}
        )
        merged = val_lgbm.merge(val_dl, on="id", how="inner")
        if len(merged) != len(tr_val):
            logger.warning(
                f"Val merge hz={hz}: dl rows {len(tr_val)} vs inner join {len(merged)}; "
                "optimizing blend on intersection only."
            )
        meta = tr_val.set_index("id").reindex(merged["id"])
        y_m = meta["y_target"].values.astype(np.float64)
        w_m = meta["weight"].values.astype(np.float64)
        p_l = merged["pred_lgbm"].values.astype(np.float64)
        p_d = merged["pred_dl"].values.astype(np.float64)

        alpha, blend_skill = grid_search_alpha(y_m, p_l, p_d, w_m)
        sk_lgbm = weighted_rmse_score(y_m, p_l, w_m)
        sk_dl = weighted_rmse_score(y_m, p_d, w_m)

        horizon_alphas[hz] = alpha
        horizon_blend_scores[hz] = blend_skill
        horizon_lgbm_scores[hz] = sk_lgbm
        horizon_dl_scores[hz] = sk_dl
        logger.info(
            f"  hz={hz} alpha(lgbm)={alpha:.4f} | skill blend={blend_skill:.6f} "
            f"lgbm_only={sk_lgbm:.6f} dl_only={sk_dl:.6f}"
        )

        cv_y.extend(y_m.tolist())
        blended = alpha * p_l + (1.0 - alpha) * p_d
        cv_p.extend(blended.tolist())
        cv_w.extend(w_m.tolist())
        cv_dl_only.extend(p_d.tolist())
        cv_lgbm_only.extend(p_l.tolist())

        if tst_pred_dl is not None and tst_lgbm is not None and len(tst_lgbm):
            dl_tst = pd.DataFrame({"id": te_hz["id"].values, "pred_dl": tst_pred_dl})
            base = pd.DataFrame({"id": te_hz["id"].values})
            base = base.merge(tst_lgbm, on="id", how="left")
            base = base.merge(dl_tst, on="id", how="left")
            if base["pred_lgbm"].isna().any() or base["pred_dl"].isna().any():
                logger.warning(
                    f"hz={hz} test: missing blend component; falling back to available model."
                )
                base["pred_lgbm"] = base["pred_lgbm"].fillna(base["pred_dl"])
                base["pred_dl"] = base["pred_dl"].fillna(base["pred_lgbm"])
            pred_b = (
                alpha * base["pred_lgbm"].values + (1.0 - alpha) * base["pred_dl"].values
            )
            test_outputs.append(
                pd.DataFrame({"id": base["id"].values, "prediction": pred_b})
            )

        del tr_hz
        gc.collect()

    final_blend = weighted_rmse_score(
        np.array(cv_y), np.array(cv_p), np.array(cv_w)
    )
    final_dl = weighted_rmse_score(
        np.array(cv_y), np.array(cv_dl_only), np.array(cv_w)
    )
    final_lgbm = weighted_rmse_score(
        np.array(cv_y), np.array(cv_lgbm_only), np.array(cv_w)
    )

    logger.info(f"FINAL CV (intersection rows) LGBM-only: {final_lgbm:.6f}")
    logger.info(f"FINAL CV DL-only: {final_dl:.6f}")
    logger.info(f"FINAL CV BLEND: {final_blend:.6f}")

    results_path = OUTPUTS_DIR / "lgbm_dl_blend_torch_results.txt"
    with open(results_path, "w") as f:
        f.write("EXPERIMENT 18: LGBM + DL PER-HORIZON BLEND\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"FINAL CV BLEND: {final_blend:.6f}\n")
        f.write(f"FINAL CV LGBM-only (same val rows): {final_lgbm:.6f}\n")
        f.write(f"FINAL CV DL-only (same val rows): {final_dl:.6f}\n\n")
        for hz in HORIZONS:
            if hz not in horizon_alphas:
                continue
            f.write(
                f"  Horizon {hz:2d}: alpha={horizon_alphas[hz]:.4f} "
                f"blend={horizon_blend_scores[hz]:.6f} "
                f"lgbm={horizon_lgbm_scores[hz]:.6f} dl={horizon_dl_scores[hz]:.6f}\n"
            )
        f.write(f"\nLGBM_SEEDS: {LGBM_SEEDS}\nDL_SEEDS: {DL_SEEDS}\n")
        f.write(f"VAL_THRESHOLD: {VAL_THRESHOLD}\n")

    logger.info(f"Wrote results: {results_path}")

    if test_outputs:
        submission = pd.concat(test_outputs, ignore_index=True)
        sub_path = OUTPUTS_DIR / "lgbm_dl_blend_torch_submission.csv"
        submission.to_csv(sub_path, index=False)
        logger.info(f"Wrote submission: {sub_path}")

    logger.info("PIPELINE COMPLETE.")
    return {
        "final_blend": final_blend,
        "final_dl": final_dl,
        "final_lgbm": final_lgbm,
        "horizon_alphas": horizon_alphas,
    }


if __name__ == "__main__":
    main()
