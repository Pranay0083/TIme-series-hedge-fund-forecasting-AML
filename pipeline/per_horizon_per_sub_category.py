#!/usr/bin/env python3
"""
Per-Horizon Per-Sub-Category LightGBM Pipeline

Trains a dedicated LightGBM model for every (horizon, sub_category) pair.
With 4 horizons [1, 3, 10, 25] and 5 sub-categories, this yields 5 × 4 = 20
independent models, each tuned on its own slice of the data.

Inherits all feature engineering from enhanced_lgbm_pipeline:
  - Target encoding (sub_category / sub_code)
  - Lag, rolling, EWM features for top features
  - Symbolic-regression discovered interaction features (GPlearn + PySR)
  - Cross-sectional rank features
  - L1 + L2 regularized LightGBM with early stopping and multi-seed ensemble
"""

import logging
import os
import sys
import gc
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "combined"
TRAIN_PATH = DATA_RAW_DIR / "train.parquet"
TEST_PATH  = DATA_RAW_DIR / "test.parquet"
LOGS_DIR   = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR  = PROJECT_ROOT / "models"

for d in (LOGS_DIR, OUTPUTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 10, 25]
VAL_THRESHOLD = 3500   # ts_index split: ≤ train, > validation


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_file: str = "per_horizon_per_sub_category.log") -> logging.Logger:
    """Configure logging to console and file."""
    logger = logging.getLogger("PerHzPerSubCat")
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
# WEIGHTED RMSE SCORE  (competition metric)
# ============================================================================
def weighted_rmse_score(y_target, y_pred, w) -> float:
    y_target, y_pred, w = np.array(y_target), np.array(y_pred), np.array(w)
    denom = np.sum(w * (y_target ** 2))
    if denom <= 0:
        return 0.0
    numerator = np.sum(w * ((y_target - y_pred) ** 2))
    return float(np.sqrt(1.0 - np.clip(numerator / denom, 0.0, 1.0)))


# ============================================================================
# TARGET ENCODING STATS  (computed once from train-only portion)
# ============================================================================
def compute_train_stats(train_path: Path, val_threshold: int, logger: logging.Logger) -> Dict:
    """Compute target encoding statistics from training data only."""
    logger.info("Computing target encoding statistics...")

    temp = pd.read_parquet(
        train_path,
        columns=["sub_category", "sub_code", "y_target", "ts_index"],
    )
    train_only = temp[temp.ts_index <= val_threshold]

    stats = {
        "sub_category": train_only.groupby("sub_category")["y_target"].mean().to_dict(),
        "sub_code":     train_only.groupby("sub_code")["y_target"].mean().to_dict(),
        "global_mean":  train_only["y_target"].mean(),
    }

    logger.info(f"  sub_category encodings : {len(stats['sub_category'])}")
    logger.info(f"  sub_code encodings     : {len(stats['sub_code'])}")
    logger.info(f"  global_mean            : {stats['global_mean']:.6f}")

    del temp, train_only
    gc.collect()
    return stats


# ============================================================================
# DISCOVERED INTERACTION FEATURES  (GPlearn + PySR symbolic regression)
# ============================================================================
def build_discovered_features(x: pd.DataFrame, horizon: int) -> pd.DataFrame:
    eps = 1e-7

    # ── GPlearn combined ──────────────────────────────────────────────────────
    x["gp_l_ca_t"] = x["feature_l"] * x["feature_ca"] * x["feature_t"]
    x["gp_ca_t"]   = x["feature_ca"] * x["feature_t"]

    # ── PySR combined ─────────────────────────────────────────────────────────
    x["psr_bz_t2"]        = x["feature_bz"] * (x["feature_t"] ** 2)
    x["psr_bz_div_bp_t2"] = x["feature_bz"] / (x["feature_bp"] / (x["feature_t"] ** 2 + eps) + eps)
    x["psr_vu_abs"]       = np.abs(x["feature_v"] * x["feature_u"])
    x["psr_bz_t"]         = x["feature_bz"] * x["feature_t"]
    x["psr_neg_s"]        = -x["feature_s"]
    x["psr_u_scaled"]     = x["feature_u"] * -0.07338854

    # ── Horizon 1 ─────────────────────────────────────────────────────────────
    x["h1_cd_div_c_q"] = (x["feature_cd"] / (x["feature_c"] + eps)) * x["feature_q"]
    x["h1_aw_inv"]     = -7.465855 / (2.7417014 - x["feature_aw"] + eps)
    x["h1_d_s"]        = x["feature_d"] * x["feature_s"]
    x["h1_abs_bz"]     = np.abs(x["feature_bz"])

    # ── Horizon 3 ─────────────────────────────────────────────────────────────
    x["h3_bz_triple"]       = 3 * x["feature_bz"]
    x["h3_bz_f_ratio"]      = x["feature_bz"] + x["feature_bz"] / (0.87934214 - x["feature_f"] + eps)
    x["h3_f_bz"]            = (x["feature_f"] - 7.829302) * x["feature_bz"]
    x["h3_bz_f_minus_b_bz"] = (x["feature_bz"] * x["feature_f"]) - (x["feature_b"] * x["feature_bz"])

    # ── Horizon 10 ────────────────────────────────────────────────────────────
    x["h10_ah_w_bv"] = x["feature_ah"] + x["feature_w"] * x["feature_bv"]
    x["h10_bs_x_aa"] = (x["feature_bs"] + x["feature_x"]) * x["feature_aa"]
    x["h10_ar_ch"]   = x["feature_ar"] - x["feature_ch"]
    x["h10_complex"] = (x["h10_ah_w_bv"] + x["h10_bs_x_aa"] + x["h10_ar_ch"]) * x["feature_bz"]
    x["h10_z_bz2"]   = x["feature_z"] * (x["feature_bz"] ** 2)
    x["h10_bz_abs_bz"] = x["feature_bz"] * np.abs(x["feature_bz"])
    x["h10_bz_s"]    = x["feature_bz"] * x["feature_s"]
    x["h10_bm_scaled"] = x["feature_bm"] * -0.0021058018

    # ── Horizon 25 ────────────────────────────────────────────────────────────
    x["h25_b_bz"]       = x["feature_b"]  * x["feature_bz"]
    x["h25_bx_bz"]      = x["feature_bx"] * x["feature_bz"]
    x["h25_n_bz"]       = x["feature_n"]  * x["feature_bz"]
    x["h25_o_bz"]       = x["feature_o"]  + x["feature_bz"]
    x["h25_s_bz_shift"] = x["feature_s"]  * (x["feature_bz"] + 1.4804125)
    x["h25_e_bz"]       = x["feature_e"]  * x["feature_bz"]
    x["h25_cd_abs_bz"]  = np.abs(x["feature_cd"]) * x["feature_bz"]

    # ── Additional polynomial / ratio features ────────────────────────────────
    x["bz_squared"]  = x["feature_bz"] ** 2
    x["bz_cubed"]    = x["feature_bz"] ** 3
    x["s_squared"]   = x["feature_s"]  ** 2
    x["t_squared"]   = x["feature_t"]  ** 2
    x["al_div_am"]   = x["feature_al"] / (x["feature_am"] + eps)
    x["bz_div_s"]    = x["feature_bz"] / (x["feature_s"]  + eps)
    x["cd_div_bz"]   = x["feature_cd"] / (x["feature_bz"] + eps)
    x["bz_div_bp"]   = x["feature_bz"] / (x["feature_bp"] + eps)

    # ── Cross-horizon interactions ────────────────────────────────────────────
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
    """Target encoding, lags, rolling, EWM, rank, and discovered interactions."""
    x = data.copy()
    group_cols  = ["code", "sub_code", "sub_category", "horizon"]
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

    # Discovered interaction features
    x = build_discovered_features(x, horizon)

    return x


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("PER-HORIZON PER-SUB-CATEGORY LIGHTGBM PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Start time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Horizons        : {HORIZONS}")
    logger.info(f"Val threshold   : ts_index > {VAL_THRESHOLD}")

    # ── LightGBM hyper-parameters ─────────────────────────────────────────────
    lgb_cfg = {
        "objective":        "regression",
        "metric":           "rmse",
        "learning_rate":    0.02,
        "n_estimators":     6000,
        "num_leaves":       96,
        "min_child_samples": 150,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "lambda_l1":        0.2,   # L1 regularisation
        "lambda_l2":        15.0,  # L2 regularisation
        "verbosity":        -1,
    }
    seeds = [42, 2024, 7, 11, 999]

    logger.info(f"LGB config      : lr={lgb_cfg['learning_rate']}, "
                f"L1={lgb_cfg['lambda_l1']}, L2={lgb_cfg['lambda_l2']}")
    logger.info(f"Ensemble seeds  : {seeds}")

    # ── Target encoding stats (once) ─────────────────────────────────────────
    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)

    # ── Discover sub_category values ─────────────────────────────────────────
    _meta = pd.read_parquet(TRAIN_PATH, columns=["sub_category"])
    sub_categories: List[str] = sorted(_meta["sub_category"].unique().tolist())
    del _meta
    gc.collect()

    logger.info(f"Sub-categories  : {sub_categories}  (n={len(sub_categories)})")

    if len(sub_categories) != 5:
        logger.warning(
            f"Expected 5 sub_categories but found {len(sub_categories)}. "
            "Proceeding with discovered values."
        )

    # ── Accumulation buffers ─────────────────────────────────────────────────
    cv_cache   = {"y": [], "pred": [], "wt": []}
    test_outputs: List[pd.DataFrame] = []

    # Scores keyed as (horizon, sub_category)
    pair_scores: Dict = {}
    feature_importance_all: Dict = {}

    # ── Main double loop: horizon × sub_category ──────────────────────────────
    for hz in HORIZONS:
        logger.info("\n" + "=" * 70)
        logger.info(f"HORIZON = {hz}")
        logger.info("=" * 70)

        # Load & feature-engineer the full horizon slice once
        logger.info(f"  Loading horizon {hz} train data...")
        hz_train = pd.read_parquet(TRAIN_PATH)
        hz_train = hz_train[hz_train["horizon"] == hz].copy()
        logger.info(f"  Horizon {hz} train rows: {len(hz_train):,}")

        hz_train = build_context_features(hz_train, train_stats, hz, logger)

        # Load test slice for this horizon (if available)
        has_test = False
        hz_test: Optional[pd.DataFrame] = None
        if TEST_PATH.exists():
            hz_test = pd.read_parquet(TEST_PATH)
            hz_test  = hz_test[hz_test["horizon"] == hz].copy()
            hz_test  = build_context_features(hz_test, train_stats, hz, logger)
            has_test = True
            logger.info(f"  Horizon {hz} test rows : {len(hz_test):,}")

        # Exclude meta-columns from features
        exclude_cols = {
            "id", "code", "sub_code", "sub_category",
            "horizon", "ts_index", "weight", "y_target",
        }
        feature_cols = [c for c in hz_train.columns if c not in exclude_cols]
        logger.info(f"  Feature columns: {len(feature_cols)}")

        # ── Per sub_category loop ─────────────────────────────────────────────
        for sc in sub_categories:
            logger.info(f"\n  ── Horizon {hz} | sub_category = '{sc}' ──")

            # Slice this sub_category
            sc_train = hz_train[hz_train["sub_category"] == sc].copy()
            logger.info(f"    Total rows (train+val): {len(sc_train):,}")

            if len(sc_train) == 0:
                logger.warning(f"    No data for hz={hz}, sc={sc}. Skipping.")
                continue

            # Time-based split
            fit_mask  = sc_train["ts_index"] <= VAL_THRESHOLD
            val_mask  = sc_train["ts_index"] >  VAL_THRESHOLD

            X_fit  = sc_train.loc[fit_mask,  feature_cols]
            y_fit  = sc_train.loc[fit_mask,  "y_target"]
            w_fit  = sc_train.loc[fit_mask,  "weight"]

            X_hold = sc_train.loc[val_mask,  feature_cols]
            y_hold = sc_train.loc[val_mask,  "y_target"]
            w_hold = sc_train.loc[val_mask,  "weight"]

            logger.info(f"    Train: {len(X_fit):,} rows | Val: {len(X_hold):,} rows")

            if len(X_fit) == 0 or len(X_hold) == 0:
                logger.warning("    Insufficient train/val rows. Skipping.")
                continue

            # Test slice for this sub_category
            sc_test: Optional[pd.DataFrame] = None
            if has_test and hz_test is not None:
                sc_test = hz_test[hz_test["sub_category"] == sc]

            # Multi-seed ensemble
            val_pred = np.zeros(len(y_hold))
            tst_pred = np.zeros(len(sc_test)) if sc_test is not None and len(sc_test) > 0 else None

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

                if tst_pred is not None and sc_test is not None and len(sc_test) > 0:
                    tst_pred += mdl.predict(sc_test[feature_cols]) / len(seeds)

                fi_accum += mdl.feature_importances_ / len(seeds)
                logger.info(f"      Best iteration: {mdl.best_iteration_}")

            # Feature importance
            imp_df = pd.DataFrame(
                {"feature": feature_cols, "importance": fi_accum}
            ).sort_values("importance", ascending=False)
            feature_importance_all[(hz, sc)] = imp_df

            zeroed = imp_df[imp_df["importance"] == 0]
            logger.info(f"    L1 zeroed features : {len(zeroed)}")
            logger.info(f"    Top-10 features    : {imp_df.head(10)['feature'].tolist()}")

            # CV accumulation
            cv_cache["y"].extend(y_hold.tolist())
            cv_cache["pred"].extend(val_pred.tolist())
            cv_cache["wt"].extend(w_hold.tolist())

            # Per-pair score
            score = weighted_rmse_score(y_hold, val_pred, w_hold)
            pair_scores[(hz, sc)] = score
            logger.info(f"    Score (hz={hz}, sc='{sc}'): {score:.5f}")

            # Test predictions
            if tst_pred is not None and sc_test is not None and len(sc_test) > 0:
                test_outputs.append(
                    pd.DataFrame({"id": sc_test["id"].values, "prediction": tst_pred})
                )

            del sc_train
            gc.collect()

        # Free horizon-level data
        del hz_train
        if hz_test is not None:
            del hz_test
        gc.collect()

    # ── Final results ─────────────────────────────────────────────────────────
    final_score = weighted_rmse_score(cv_cache["y"], cv_cache["pred"], cv_cache["wt"])

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"FINAL AGGREGATE SCORE: {final_score:.6f}")
    logger.info("\nPer-Horizon Aggregate Scores:")
    for hz in HORIZONS:
        hz_pairs = {sc: s for (h, sc), s in pair_scores.items() if h == hz}
        if hz_pairs:
            avg = np.mean(list(hz_pairs.values()))
            logger.info(f"  Horizon {hz:2d} average: {avg:.5f}")
            for sc, s in hz_pairs.items():
                logger.info(f"    sub_category='{sc}': {s:.5f}")

    # ── Save results text ─────────────────────────────────────────────────────
    results_path = OUTPUTS_DIR / "per_horizon_per_sub_category_results.txt"
    with open(results_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PER-HORIZON PER-SUB-CATEGORY LIGHTGBM PIPELINE\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"FINAL AGGREGATE SCORE: {final_score:.6f}\n\n")
        f.write(f"Validation threshold  : ts_index > {VAL_THRESHOLD}\n")
        f.write(f"Ensemble seeds        : {seeds}\n")
        f.write(f"L1 regularization     : {lgb_cfg['lambda_l1']}\n")
        f.write(f"L2 regularization     : {lgb_cfg['lambda_l2']}\n\n")

        f.write("Per-(horizon, sub_category) scores:\n")
        for hz in HORIZONS:
            f.write(f"\n  Horizon {hz}:\n")
            for (h, sc), score in pair_scores.items():
                if h == hz:
                    f.write(f"    sub_category='{sc}': {score:.5f}\n")

        # Top-20 features per model
        for (hz, sc), imp_df in feature_importance_all.items():
            f.write(f"\n{'='*40}\n")
            f.write(f"HORIZON {hz} | sub_category='{sc}' — TOP 20 FEATURES\n")
            f.write(f"{'='*40}\n")
            for _, row in imp_df.head(20).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.2f}\n")
            zeroed = imp_df[imp_df["importance"] == 0]
            f.write(f"\nL1 zeroed features: {len(zeroed)}\n")

    logger.info(f"Results saved to: {results_path}")

    # ── Save submission ───────────────────────────────────────────────────────
    if test_outputs:
        submission = pd.concat(test_outputs).sort_values("id").reset_index(drop=True)
        submission_path = OUTPUTS_DIR / "per_horizon_per_sub_category_submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to: {submission_path}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)

    return {"final_score": final_score, "pair_scores": pair_scores}


if __name__ == "__main__":
    main()
