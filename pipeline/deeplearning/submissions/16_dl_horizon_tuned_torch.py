#!/usr/bin/env python3
"""
Experiment 16 — Horizon-conditioned PyTorch capacity

Hypothesis
    Short horizons (1, 3) need a smaller latent and heavier dropout to limit
    overfitting; long horizons (10, 25) need more capacity and training budget.
    This is the same structural idea as different `num_leaves` by horizon in
    05_estimator_architectures, transferred to the MLP.

What changes vs Experiment 15
    Per-horizon overrides on `latent_dim`, `drop_rate`, `patience`, and
    `max_epochs` for horizons 10 and 25. Everything else matches the baseline.

What is held fixed
    Paths, VAL_THRESHOLD, feature pipeline, single seed 42, metric, clipping.

Outputs
    outputs/dl_horizon_tuned_torch_results.txt
    outputs/dl_horizon_tuned_torch_submission.csv

Risks
    Stronger long-horizon models can increase variance; monitor CV aggregate.
"""
import importlib.util
import os
import sys
from datetime import datetime

import gc
import numpy as np
import pandas as pd

_DL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DL_ROOT not in sys.path:
    sys.path.append(_DL_ROOT)


def import_local_module(module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_DL_ROOT, f"{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod_01_paths = import_local_module("01_paths")
mod_02_log = import_local_module("02_logging")
mod_obj = import_local_module("01_optimization_objective")
mod_03 = import_local_module("03_feature_engineering")
mod_07 = import_local_module("07_post_processing_ensembling")
mod_08 = import_local_module("08_pipeline_infrastructure")
mod_09 = import_local_module("09_torch_estimator")

TRAIN_PATH = mod_01_paths.TRAIN_PATH
TEST_PATH = mod_01_paths.TEST_PATH
OUTPUTS_DIR = mod_01_paths.OUTPUTS_DIR
HORIZONS = mod_01_paths.HORIZONS
VAL_THRESHOLD = mod_01_paths.VAL_THRESHOLD
DL_MODELS_DIR = mod_01_paths.DL_MODELS_DIR

setup_logging = mod_02_log.setup_logging
weighted_rmse_score = mod_obj.custom_weighted_rmse_score
create_spread_ratios = mod_03.create_spread_ratios
create_group_z_scores = mod_03.create_group_z_scores
create_rolling_lag_features = mod_03.create_rolling_lag_features
ExpandingMeanTargetEncoder = mod_03.ExpandingMeanTargetEncoder
apply_target_clipping = mod_07.apply_target_clipping
aggressive_downcasting = mod_08.aggressive_downcasting
MemoryMonitor = mod_08.MemoryMonitor
HorizonTorchEstimator = mod_09.HorizonTorchEstimator

EXCLUDE_COLS = {
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


def main():
    logger = setup_logging("DLHorizonTunedTorch", "16_dl_horizon_tuned_torch.log")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 16: HORIZON-TUNED PYTORCH MLP")
    logger.info("=" * 60)
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Train: {TRAIN_PATH}")
    logger.info(f"Test:  {TEST_PATH} (exists={TEST_PATH.exists()})")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Validation holdout: ts_index > {VAL_THRESHOLD}")
    logger.info(f"Horizon overrides: {HORIZON_OVERRIDES}")

    base_torch_kw = dict(
        latent_dim=32,
        drop_rate=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=2048,
        max_epochs=200,
        patience=25,
        val_frac_timesteps=0.12,
    )
    seeds = [42]
    logger.info(f"Base torch kwargs: {base_torch_kw}")
    logger.info(f"Seeds: {seeds}")

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

    cv_cache = {"y": [], "pred": [], "wt": []}
    test_outputs = []
    horizon_scores = {}
    feature_cols_last = []

    for hz in HORIZONS:
        logger.info("\n" + "=" * 60)
        logger.info(f"HORIZON = {hz}")
        logger.info("=" * 60)

        tr_hz = train_fe[train_fe["horizon"] == hz].copy()
        logger.info(f"Rows (train, horizon {hz}): {len(tr_hz):,}")

        if has_test:
            te_hz = test_fe[test_fe["horizon"] == hz].copy()
            logger.info(f"Rows (test, horizon {hz}): {len(te_hz):,}")
        else:
            te_hz = None

        fit_mask = tr_hz.ts_index <= VAL_THRESHOLD
        val_mask = tr_hz.ts_index > VAL_THRESHOLD
        tr_fit = tr_hz.loc[fit_mask]
        tr_val = tr_hz.loc[val_mask]

        feature_cols = [
            c
            for c in tr_hz.columns
            if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(tr_hz[c])
        ]
        feature_cols_last = feature_cols

        y_hold = tr_val["y_target"].values
        w_hold = tr_val["weight"].values
        val_pred = np.zeros(len(tr_val))
        tst_pred = None
        if has_test and len(te_hz):
            tst_pred = np.zeros(len(te_hz))

        torch_kw = {**base_torch_kw, **HORIZON_OVERRIDES.get(hz, {})}
        logger.info(f"Effective torch kwargs: {torch_kw}")

        for seed in seeds:
            logger.info(f"  Training seed {seed}...")
            est = HorizonTorchEstimator(horizon=hz, random_seed=seed, **torch_kw)
            est.fit(
                tr_fit,
                tr_fit["y_target"].values,
                tr_fit["weight"].values if "weight" in tr_fit.columns else None,
                X_val=tr_val,
                y_val=tr_val["y_target"].values,
                w_val=tr_val["weight"].values if "weight" in tr_val.columns else None,
            )
            val_pred += est.predict(tr_val) / len(seeds)
            if tst_pred is not None:
                raw = est.predict(te_hz)
                clipped = apply_target_clipping(raw, tr_fit["y_target"].values)
                tst_pred += clipped / len(seeds)
            ckpt = DL_MODELS_DIR / f"16_dl_horizon_tuned_torch_horizon_{hz}_seed_{seed}.pt"
            DL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            est.save(str(ckpt))

        cv_cache["y"].extend(y_hold.tolist())
        cv_cache["pred"].extend(val_pred.tolist())
        cv_cache["wt"].extend(w_hold.tolist())

        hz_score = weighted_rmse_score(y_hold, val_pred, sample_weight=w_hold)
        horizon_scores[hz] = hz_score
        logger.info(f"Horizon {hz} CV skill: {hz_score:.6f}")

        if tst_pred is not None:
            test_outputs.append(pd.DataFrame({"id": te_hz["id"], "prediction": tst_pred}))

        del tr_hz
        gc.collect()

    final_score = weighted_rmse_score(
        np.array(cv_cache["y"]),
        np.array(cv_cache["pred"]),
        sample_weight=np.array(cv_cache["wt"]),
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL AGGREGATE CV SKILL: {final_score:.6f}")

    results_path = OUTPUTS_DIR / "dl_horizon_tuned_torch_results.txt"
    with open(results_path, "w") as f:
        f.write("EXPERIMENT 16: HORIZON-TUNED PYTORCH\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"FINAL AGGREGATE CV SKILL: {final_score:.6f}\n\n")
        for hz, score in horizon_scores.items():
            f.write(f"  Horizon {hz:2d}: {score:.6f}\n")
        f.write(f"\nVAL_THRESHOLD: {VAL_THRESHOLD}\n")
        f.write(f"Overrides: {HORIZON_OVERRIDES}\n")
        f.write(f"Features (last hz): {len(feature_cols_last)}\n")

    logger.info(f"Wrote results: {results_path}")

    if test_outputs:
        submission = pd.concat(test_outputs, ignore_index=True)
        sub_path = OUTPUTS_DIR / "dl_horizon_tuned_torch_submission.csv"
        submission.to_csv(sub_path, index=False)
        logger.info(f"Wrote submission: {sub_path}")

    logger.info("PIPELINE COMPLETE.")
    return {"final_score": final_score, "horizon_scores": horizon_scores}


if __name__ == "__main__":
    main()
