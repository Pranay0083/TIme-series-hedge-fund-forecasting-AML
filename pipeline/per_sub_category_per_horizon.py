#!/usr/bin/env python3
"""
Per-Sub-Category Per-Horizon LightGBM Pipeline

Trains a dedicated LightGBM model for every (sub_category, horizon) pair.
With 5 sub-categories and 4 horizons [1, 3, 10, 25] this yields 5 × 4 = 20
independent models.

Loop order  →  outer: sub_category  |  inner: horizon
(Compare to per_horizon_per_sub_category.py which does outer: horizon, inner: sub_category)

All feature engineering is inherited from enhanced_lgbm_pipeline:
  - Target encoding (sub_category / sub_code)
  - Lag / rolling / EWM features for top features
  - GPlearn + PySR discovered interaction features
  - Cross-sectional rank features
  - L1 + L2 regularised LightGBM, multi-seed ensemble, early stopping
"""

import logging
import os
import sys
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "combined"
TRAIN_PATH   = DATA_RAW_DIR / "train.parquet"
TEST_PATH    = DATA_RAW_DIR / "test.parquet"
LOGS_DIR     = PROJECT_ROOT / "logs"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
MODELS_DIR   = PROJECT_ROOT / "models"

for _d in (LOGS_DIR, OUTPUTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HORIZONS      = [1, 3, 10, 25]
VAL_THRESHOLD = 3500   # ts_index ≤ train, > validation


# ============================================================================
# LOGGING
# ============================================================================
def setup_logging(log_file: str = "per_sub_category_per_horizon.log") -> logging.Logger:
    logger = logging.getLogger("PerSubCatPerHz")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOGS_DIR / log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================================
# WEIGHTED RMSE  (competition metric)
# ============================================================================
def weighted_rmse_score(y_target, y_pred, w) -> float:
    y_target, y_pred, w = np.array(y_target), np.array(y_pred), np.array(w)
    denom = np.sum(w * (y_target ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(1.0 - np.clip(np.sum(w * (y_target - y_pred) ** 2) / denom, 0.0, 1.0)))


# ============================================================================
# TARGET ENCODING STATS
# ============================================================================
def compute_train_stats(train_path: Path, val_threshold: int, logger: logging.Logger) -> Dict:
    logger.info("Computing target encoding statistics...")
    temp = pd.read_parquet(train_path, columns=["sub_category", "sub_code", "y_target", "ts_index"])
    tr   = temp[temp.ts_index <= val_threshold]

    stats = {
        "sub_category": tr.groupby("sub_category")["y_target"].mean().to_dict(),
        "sub_code":     tr.groupby("sub_code")["y_target"].mean().to_dict(),
        "global_mean":  tr["y_target"].mean(),
    }
    logger.info(f"  sub_category encodings : {len(stats['sub_category'])}")
    logger.info(f"  sub_code encodings     : {len(stats['sub_code'])}")
    logger.info(f"  global_mean            : {stats['global_mean']:.6f}")

    del temp, tr
    gc.collect()
    return stats


# ============================================================================
# DISCOVERED INTERACTION FEATURES  (GPlearn + PySR)
# ============================================================================
def build_discovered_features(x: pd.DataFrame, horizon: int) -> pd.DataFrame:
    eps = 1e-7

    # GPlearn combined
    x["gp_l_ca_t"] = x["feature_l"] * x["feature_ca"] * x["feature_t"]
    x["gp_ca_t"]   = x["feature_ca"] * x["feature_t"]

    # PySR combined
    x["psr_bz_t2"]        = x["feature_bz"] * (x["feature_t"] ** 2)
    x["psr_bz_div_bp_t2"] = x["feature_bz"] / (x["feature_bp"] / (x["feature_t"] ** 2 + eps) + eps)
    x["psr_vu_abs"]       = np.abs(x["feature_v"] * x["feature_u"])
    x["psr_bz_t"]         = x["feature_bz"] * x["feature_t"]
    x["psr_neg_s"]        = -x["feature_s"]
    x["psr_u_scaled"]     = x["feature_u"] * -0.07338854

    # Horizon 1
    x["h1_cd_div_c_q"] = (x["feature_cd"] / (x["feature_c"] + eps)) * x["feature_q"]
    x["h1_aw_inv"]     = -7.465855 / (2.7417014 - x["feature_aw"] + eps)
    x["h1_d_s"]        = x["feature_d"] * x["feature_s"]
    x["h1_abs_bz"]     = np.abs(x["feature_bz"])

    # Horizon 3
    x["h3_bz_triple"]       = 3 * x["feature_bz"]
    x["h3_bz_f_ratio"]      = x["feature_bz"] + x["feature_bz"] / (0.87934214 - x["feature_f"] + eps)
    x["h3_f_bz"]            = (x["feature_f"] - 7.829302) * x["feature_bz"]
    x["h3_bz_f_minus_b_bz"] = (x["feature_bz"] * x["feature_f"]) - (x["feature_b"] * x["feature_bz"])

    # Horizon 10
    x["h10_ah_w_bv"]   = x["feature_ah"] + x["feature_w"] * x["feature_bv"]
    x["h10_bs_x_aa"]   = (x["feature_bs"] + x["feature_x"]) * x["feature_aa"]
    x["h10_ar_ch"]     = x["feature_ar"] - x["feature_ch"]
    x["h10_complex"]   = (x["h10_ah_w_bv"] + x["h10_bs_x_aa"] + x["h10_ar_ch"]) * x["feature_bz"]
    x["h10_z_bz2"]     = x["feature_z"] * (x["feature_bz"] ** 2)
    x["h10_bz_abs_bz"] = x["feature_bz"] * np.abs(x["feature_bz"])
    x["h10_bz_s"]      = x["feature_bz"] * x["feature_s"]
    x["h10_bm_scaled"] = x["feature_bm"] * -0.0021058018

    # Horizon 25
    x["h25_b_bz"]       = x["feature_b"]  * x["feature_bz"]
    x["h25_bx_bz"]      = x["feature_bx"] * x["feature_bz"]
    x["h25_n_bz"]       = x["feature_n"]  * x["feature_bz"]
    x["h25_o_bz"]       = x["feature_o"]  + x["feature_bz"]
    x["h25_s_bz_shift"] = x["feature_s"]  * (x["feature_bz"] + 1.4804125)
    x["h25_e_bz"]       = x["feature_e"]  * x["feature_bz"]
    x["h25_cd_abs_bz"]  = np.abs(x["feature_cd"]) * x["feature_bz"]

    # Polynomial / ratio
    x["bz_squared"] = x["feature_bz"] ** 2
    x["bz_cubed"]   = x["feature_bz"] ** 3
    x["s_squared"]  = x["feature_s"]  ** 2
    x["t_squared"]  = x["feature_t"]  ** 2
    x["al_div_am"]  = x["feature_al"] / (x["feature_am"] + eps)
    x["bz_div_s"]   = x["feature_bz"] / (x["feature_s"]  + eps)
    x["cd_div_bz"]  = x["feature_cd"] / (x["feature_bz"] + eps)
    x["bz_div_bp"]  = x["feature_bz"] / (x["feature_bp"] + eps)

    # Cross-horizon interactions
    x["al_bz"] = x["feature_al"] * x["feature_bz"]
    x["al_s"]  = x["feature_al"] * x["feature_s"]
    x["v_u"]   = x["feature_v"]  * x["feature_u"]
    x["cg_by"] = x["feature_cg"] * x["feature_by"]

    return x


# ============================================================================
# CONTEXT FEATURE ENGINEERING
# ============================================================================
def build_context_features(
    data: pd.DataFrame,
    enc_stats: Dict,
    horizon: int,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    x = data.copy()
    group_cols   = ["code", "sub_code", "sub_category", "horizon"]
    top_features = ["feature_al", "feature_am", "feature_cg", "feature_by", "feature_s", "feature_bz"]

    # Target encoding
    for c in ["sub_category", "sub_code"]:
        x[c + "_enc"] = x[c].map(enc_stats[c]).fillna(enc_stats["global_mean"])

    # Basic interactions
    x["d_al_am"] = x["feature_al"] - x["feature_am"]
    x["r_al_am"] = x["feature_al"] / (x["feature_am"] + 1e-7)
    x["d_cg_by"] = x["feature_cg"] - x["feature_by"]

    # Lag features
    for col in top_features:
        if col not in x.columns:
            continue
        for lag in [1, 3, 10]:
            x[f"{col}_lag{lag}"] = x.groupby(group_cols)[col].shift(lag).astype(np.float32)
        x[f"{col}_diff1"] = x.groupby(group_cols)[col].diff(1).astype(np.float32)

    # Rolling mean / std + EWM
    for col in top_features:
        if col not in x.columns:
            continue
        for window in [5, 10]:
            x[f"{col}_roll{window}"] = (
                x.groupby(group_cols)[col]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
                .astype(np.float32)
            )
            x[f"{col}_rollstd{window}"] = (
                x.groupby(group_cols)[col]
                .transform(lambda s: s.rolling(window, min_periods=1).std())
                .astype(np.float32)
            )
        x[f"{col}_ewm5"] = (
            x.groupby(group_cols)[col]
            .transform(lambda s: s.ewm(span=5, adjust=False).mean())
            .astype(np.float32)
        )

    # Temporal signal
    x["t_cycle"] = np.sin(2 * np.pi * x["ts_index"] / 100)

    # Cross-sectional rank
    for col in ["feature_al", "feature_am", "feature_cg", "feature_by", "d_al_am", "feature_bz"]:
        if col in x.columns:
            x[col + "_rk"] = x.groupby("ts_index")[col].rank(pct=True).astype(np.float32)

    # Discovered interactions
    x = build_discovered_features(x, horizon)

    return x


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("PER-SUB-CATEGORY PER-HORIZON LIGHTGBM PIPELINE")
    logger.info("Loop order: outer=sub_category  inner=horizon")
    logger.info("=" * 70)
    logger.info(f"Start time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Horizons        : {HORIZONS}")
    logger.info(f"Val threshold   : ts_index > {VAL_THRESHOLD}")

    # LightGBM config
    lgb_cfg = {
        "objective":         "regression",
        "metric":            "rmse",
        "learning_rate":     0.02,
        "n_estimators":      6000,
        "num_leaves":        96,
        "min_child_samples": 150,
        "feature_fraction":  0.7,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l1":         0.2,
        "lambda_l2":         15.0,
        "verbosity":         -1,
    }
    seeds = [42, 2024, 7, 11, 999]

    logger.info(f"LGB config      : lr={lgb_cfg['learning_rate']}, "
                f"L1={lgb_cfg['lambda_l1']}, L2={lgb_cfg['lambda_l2']}")
    logger.info(f"Ensemble seeds  : {seeds}")

    # Target encoding stats (computed once)
    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)

    # Discover sub_categories from train data
    _meta = pd.read_parquet(TRAIN_PATH, columns=["sub_category"])
    sub_categories: List[str] = sorted(_meta["sub_category"].unique().tolist())
    del _meta
    gc.collect()

    logger.info(f"Sub-categories  : {sub_categories}  (n={len(sub_categories)})")
    if len(sub_categories) != 5:
        logger.warning(
            f"Expected 5 sub_categories but found {len(sub_categories)}. "
            "Proceeding anyway."
        )

    # Accumulation buffers
    cv_cache   = {"y": [], "pred": [], "wt": []}
    test_outputs: List[pd.DataFrame] = []
    pair_scores: Dict             = {}
    feature_importance_all: Dict  = {}

    # ── Outer loop: sub_category ───────────────────────────────────────────────
    for sc in sub_categories:
        logger.info("\n" + "=" * 70)
        logger.info(f"SUB-CATEGORY = '{sc}'")
        logger.info("=" * 70)

        # Load full sub_category slice (all horizons) once
        logger.info(f"  Loading train data for sub_category='{sc}'...")
        sc_train_full = pd.read_parquet(TRAIN_PATH)
        sc_train_full = sc_train_full[sc_train_full["sub_category"] == sc].copy()
        logger.info(f"  Sub-category '{sc}' train rows: {len(sc_train_full):,}")

        # Load test slice for this sub_category (if available)
        has_test = False
        sc_test_full: Optional[pd.DataFrame] = None
        if TEST_PATH.exists():
            sc_test_full = pd.read_parquet(TEST_PATH)
            sc_test_full = sc_test_full[sc_test_full["sub_category"] == sc].copy()
            has_test = True
            logger.info(f"  Sub-category '{sc}' test rows : {len(sc_test_full):,}")

        # ── Inner loop: horizon ───────────────────────────────────────────────
        for hz in HORIZONS:
            logger.info(f"\n  ── sub_category='{sc}' | Horizon = {hz} ──")

            # Slice by horizon
            hz_train = sc_train_full[sc_train_full["horizon"] == hz].copy()
            logger.info(f"    Total rows (train+val): {len(hz_train):,}")

            if len(hz_train) == 0:
                logger.warning(f"    No data for sc='{sc}', hz={hz}. Skipping.")
                continue

            # Feature engineering
            hz_train = build_context_features(hz_train, train_stats, hz, logger)

            hz_test: Optional[pd.DataFrame] = None
            if has_test and sc_test_full is not None:
                hz_test = sc_test_full[sc_test_full["horizon"] == hz].copy()
                if len(hz_test) > 0:
                    hz_test = build_context_features(hz_test, train_stats, hz, logger)
                    logger.info(f"    Test rows: {len(hz_test):,}")

            # Feature columns
            exclude_cols = {
                "id", "code", "sub_code", "sub_category",
                "horizon", "ts_index", "weight", "y_target",
            }
            feature_cols = [c for c in hz_train.columns if c not in exclude_cols]
            logger.info(f"    Feature columns: {len(feature_cols)}")

            # Time-based split
            fit_mask  = hz_train["ts_index"] <= VAL_THRESHOLD
            val_mask  = hz_train["ts_index"] >  VAL_THRESHOLD

            X_fit  = hz_train.loc[fit_mask,  feature_cols]
            y_fit  = hz_train.loc[fit_mask,  "y_target"]
            w_fit  = hz_train.loc[fit_mask,  "weight"]

            X_hold = hz_train.loc[val_mask,  feature_cols]
            y_hold = hz_train.loc[val_mask,  "y_target"]
            w_hold = hz_train.loc[val_mask,  "weight"]

            logger.info(f"    Train: {len(X_fit):,} | Val: {len(X_hold):,}")

            if len(X_fit) == 0 or len(X_hold) == 0:
                logger.warning("    Insufficient train/val rows. Skipping.")
                del hz_train
                gc.collect()
                continue

            # Multi-seed ensemble
            val_pred = np.zeros(len(y_hold))
            tst_pred = (
                np.zeros(len(hz_test))
                if hz_test is not None and len(hz_test) > 0
                else None
            )
            fi_accum = np.zeros(len(feature_cols))

            for seed in seeds:
                logger.info(f"    Training seed {seed}...")
                mdl = lgb.LGBMRegressor(**lgb_cfg, random_state=seed)
                mdl.fit(
                    X_fit,
                    y_fit,
                    sample_weight=w_fit,
                    eval_set=[(X_hold, y_hold)],
                    eval_sample_weight=[w_hold],
                    callbacks=[lgb.early_stopping(200, verbose=False)],
                )

                val_pred += mdl.predict(X_hold) / len(seeds)
                if tst_pred is not None and hz_test is not None:
                    tst_pred += mdl.predict(hz_test[feature_cols]) / len(seeds)
                fi_accum += mdl.feature_importances_ / len(seeds)
                logger.info(f"      Best iteration: {mdl.best_iteration_}")

            # Feature importance
            imp_df = pd.DataFrame(
                {"feature": feature_cols, "importance": fi_accum}
            ).sort_values("importance", ascending=False)
            feature_importance_all[(sc, hz)] = imp_df

            zeroed = imp_df[imp_df["importance"] == 0]
            logger.info(f"    L1 zeroed features : {len(zeroed)}")
            logger.info(f"    Top-10 features    : {imp_df.head(10)['feature'].tolist()}")

            # CV accumulation
            cv_cache["y"].extend(y_hold.tolist())
            cv_cache["pred"].extend(val_pred.tolist())
            cv_cache["wt"].extend(w_hold.tolist())

            # Per-pair score
            score = weighted_rmse_score(y_hold, val_pred, w_hold)
            pair_scores[(sc, hz)] = score
            logger.info(f"    Score (sc='{sc}', hz={hz}): {score:.5f}")

            # Test predictions
            if tst_pred is not None and hz_test is not None and len(hz_test) > 0:
                test_outputs.append(
                    pd.DataFrame({"id": hz_test["id"].values, "prediction": tst_pred})
                )

            del hz_train
            if hz_test is not None:
                del hz_test
            gc.collect()

        # Free sub_category-level data
        del sc_train_full
        if sc_test_full is not None:
            del sc_test_full
        gc.collect()

    # ── Final results ─────────────────────────────────────────────────────────
    final_score = weighted_rmse_score(cv_cache["y"], cv_cache["pred"], cv_cache["wt"])

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"FINAL AGGREGATE SCORE: {final_score:.6f}")
    logger.info("\nPer-Sub-Category Aggregate Scores:")
    for sc in sub_categories:
        sc_pairs = {hz: s for (s_cat, hz), s in pair_scores.items() if s_cat == sc}
        if sc_pairs:
            avg = np.mean(list(sc_pairs.values()))
            logger.info(f"  sub_category='{sc}' average: {avg:.5f}")
            for hz, s in sc_pairs.items():
                logger.info(f"    Horizon {hz:2d}: {s:.5f}")

    # Save results
    results_path = OUTPUTS_DIR / "per_sub_category_per_horizon_results.txt"
    with open(results_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PER-SUB-CATEGORY PER-HORIZON LIGHTGBM PIPELINE\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"FINAL AGGREGATE SCORE: {final_score:.6f}\n\n")
        f.write(f"Validation threshold : ts_index > {VAL_THRESHOLD}\n")
        f.write(f"Ensemble seeds       : {seeds}\n")
        f.write(f"L1 regularization    : {lgb_cfg['lambda_l1']}\n")
        f.write(f"L2 regularization    : {lgb_cfg['lambda_l2']}\n\n")

        f.write("Per-(sub_category, horizon) scores:\n")
        for sc in sub_categories:
            f.write(f"\n  sub_category='{sc}':\n")
            for (s_cat, hz), score in pair_scores.items():
                if s_cat == sc:
                    f.write(f"    Horizon {hz:2d}: {score:.5f}\n")

        for (sc, hz), imp_df in feature_importance_all.items():
            f.write(f"\n{'='*40}\n")
            f.write(f"sub_category='{sc}' | HORIZON {hz} — TOP 20 FEATURES\n")
            f.write(f"{'='*40}\n")
            for _, row in imp_df.head(20).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.2f}\n")
            zeroed = imp_df[imp_df["importance"] == 0]
            f.write(f"\nL1 zeroed features: {len(zeroed)}\n")

    logger.info(f"Results saved to: {results_path}")

    # Save submission
    if test_outputs:
        submission = pd.concat(test_outputs).sort_values("id").reset_index(drop=True)
        submission_path = OUTPUTS_DIR / "per_sub_category_per_horizon_submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to: {submission_path}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)

    return {"final_score": final_score, "pair_scores": pair_scores}


if __name__ == "__main__":
    main()
