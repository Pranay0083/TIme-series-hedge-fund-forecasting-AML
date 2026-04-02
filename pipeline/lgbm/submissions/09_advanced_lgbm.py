#!/usr/bin/env python3
"""Advanced LightGBM horizon pipeline — entry script."""
import importlib.util
import os
import sys
from datetime import datetime

import gc
import lightgbm as lgb
import numpy as np
import pandas as pd

# ============================================================================
# LOCAL MODULE LOADER (shared modules in parent pipeline/lgbm/)
# ============================================================================
_LGBM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LGBM_ROOT not in sys.path:
    sys.path.append(_LGBM_ROOT)


def import_local_module(module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_LGBM_ROOT, f"{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod_01 = import_local_module("01_paths")
mod_02 = import_local_module("02_logging")
mod_03 = import_local_module("03_metrics")
mod_04 = import_local_module("04_encoding_stats")
mod_05 = import_local_module("05_features_advanced")

TRAIN_PATH = mod_01.TRAIN_PATH
TEST_PATH = mod_01.TEST_PATH
OUTPUTS_DIR = mod_01.OUTPUTS_DIR
HORIZONS = mod_01.HORIZONS
VAL_THRESHOLD = mod_01.VAL_THRESHOLD
setup_logging = mod_02.setup_logging
weighted_rmse_score = mod_03.weighted_rmse_score
compute_train_stats = mod_04.compute_train_stats
build_context_features = mod_05.build_context_features


def main():
    logger = setup_logging("AdvancedLGBM", "advanced_lgbm_pipeline.log")

    logger.info("=" * 60)
    logger.info("ADVANCED LIGHTGBM HORIZON PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Validation threshold: ts_index > {VAL_THRESHOLD}")

    lgb_cfg = {
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

    logger.info(f"LightGBM config: lr={lgb_cfg['learning_rate']}, leaves={lgb_cfg['num_leaves']}")

    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)
    cv_cache = {"y": [], "pred": [], "wt": []}
    test_outputs = []
    horizon_scores = {}
    seeds = [42, 2024, 7, 11, 999]
    logger.info(f"Ensemble seeds: {seeds}")

    for hz in HORIZONS:
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING HORIZON = {hz}")
        logger.info("=" * 60)

        logger.info(f"Loading horizon {hz} data...")
        tr_df = pd.read_parquet(TRAIN_PATH)
        tr_df = tr_df[tr_df["horizon"] == hz].copy()
        logger.info(f"Train data: {len(tr_df):,} rows")

        logger.info("Building context features...")
        tr_df = build_context_features(tr_df, train_stats, logger)

        if TEST_PATH.exists():
            te_df = pd.read_parquet(TEST_PATH)
            te_df = te_df[te_df["horizon"] == hz].copy()
            te_df = build_context_features(te_df, train_stats, logger)
            has_test = True
            logger.info(f"Test data: {len(te_df):,} rows")
        else:
            has_test = False
            logger.info("No test file found, skipping test predictions")

        exclude_cols = {
            "id", "code", "sub_code", "sub_category",
            "horizon", "ts_index", "weight", "y_target",
        }
        feature_cols = [c for c in tr_df.columns if c not in exclude_cols]
        logger.info(f"Feature columns: {len(feature_cols)}")

        fit_mask = tr_df.ts_index <= VAL_THRESHOLD
        val_mask = tr_df.ts_index > VAL_THRESHOLD

        X_fit = tr_df.loc[fit_mask, feature_cols]
        y_fit = tr_df.loc[fit_mask, "y_target"]
        w_fit = tr_df.loc[fit_mask, "weight"]

        X_hold = tr_df.loc[val_mask, feature_cols]
        y_hold = tr_df.loc[val_mask, "y_target"]
        w_hold = tr_df.loc[val_mask, "weight"]

        logger.info(f"Train: {len(X_fit):,} rows (ts_index <= {VAL_THRESHOLD})")
        logger.info(f"Val:   {len(X_hold):,} rows (ts_index > {VAL_THRESHOLD})")

        val_pred = np.zeros(len(y_hold))
        if has_test:
            tst_pred = np.zeros(len(te_df))

        for seed in seeds:
            logger.info(f"  Training seed {seed}...")
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
            if has_test:
                tst_pred += mdl.predict(te_df[feature_cols]) / len(seeds)
            logger.info(f"    Best iteration: {mdl.best_iteration_}")

        cv_cache["y"].extend(y_hold.tolist())
        cv_cache["pred"].extend(val_pred.tolist())
        cv_cache["wt"].extend(w_hold.tolist())

        hz_score = weighted_rmse_score(y_hold, val_pred, w_hold)
        horizon_scores[hz] = hz_score
        logger.info(f"\nHorizon {hz} Score: {hz_score:.5f}")

        if has_test:
            test_outputs.append(pd.DataFrame({"id": te_df["id"], "prediction": tst_pred}))

        del tr_df
        if has_test:
            del te_df
        gc.collect()

    final_score = weighted_rmse_score(cv_cache["y"], cv_cache["pred"], cv_cache["wt"])

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"FINAL AGGREGATE SCORE: {final_score:.6f}")
    logger.info("\nPer-Horizon Scores:")
    for hz, score in horizon_scores.items():
        logger.info(f"  Horizon {hz:2d}: {score:.5f}")

    results_path = OUTPUTS_DIR / "advanced_lgbm_results.txt"
    with open(results_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ADVANCED LIGHTGBM PIPELINE RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"FINAL AGGREGATE SCORE: {final_score:.6f}\n\n")
        f.write("Per-Horizon Scores:\n")
        for hz, score in horizon_scores.items():
            f.write(f"  Horizon {hz:2d}: {score:.5f}\n")
        f.write("\n")
        f.write(f"Validation threshold: ts_index > {VAL_THRESHOLD}\n")
        f.write(f"Ensemble seeds: {seeds}\n")
        f.write(f"Total features: {len(feature_cols)}\n")

    logger.info(f"Results saved to: {results_path}")

    if test_outputs:
        submission = pd.concat(test_outputs)
        submission_path = OUTPUTS_DIR / "submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to: {submission_path}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {"final_score": final_score, "horizon_scores": horizon_scores}


if __name__ == "__main__":
    main()
