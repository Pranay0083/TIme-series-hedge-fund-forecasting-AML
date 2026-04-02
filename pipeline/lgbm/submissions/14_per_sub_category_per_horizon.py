#!/usr/bin/env python3
"""Per-sub-category × per-horizon LightGBM — entry script."""
import importlib.util
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import gc
import lightgbm as lgb
import numpy as np
import pandas as pd

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
mod_06 = import_local_module("06_features_enhanced")

TRAIN_PATH = mod_01.TRAIN_PATH
TEST_PATH = mod_01.TEST_PATH
OUTPUTS_DIR = mod_01.OUTPUTS_DIR
HORIZONS = mod_01.HORIZONS
VAL_THRESHOLD = mod_01.VAL_THRESHOLD
setup_logging = mod_02.setup_logging
weighted_rmse_score = mod_03.weighted_rmse_score
compute_train_stats = mod_04.compute_train_stats
build_context_features = mod_06.build_context_features


def main():
    logger = setup_logging("PerSubCatPerHz", "per_sub_category_per_horizon.log")

    logger.info("=" * 70)
    logger.info("PER-SUB-CATEGORY PER-HORIZON LIGHTGBM PIPELINE")
    logger.info("Loop order: outer=sub_category  inner=horizon")
    logger.info("=" * 70)
    logger.info(f"Start time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Horizons        : {HORIZONS}")
    logger.info(f"Val threshold   : ts_index > {VAL_THRESHOLD}")

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
    seeds = [42, 2024, 7, 11, 999]

    logger.info(
        f"LGB config      : lr={lgb_cfg['learning_rate']}, "
        f"L1={lgb_cfg['lambda_l1']}, L2={lgb_cfg['lambda_l2']}"
    )
    logger.info(f"Ensemble seeds  : {seeds}")

    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)

    _meta = pd.read_parquet(TRAIN_PATH, columns=["sub_category"])
    sub_categories: List = sorted(_meta["sub_category"].unique().tolist())
    del _meta
    gc.collect()

    logger.info(f"Sub-categories  : {sub_categories}  (n={len(sub_categories)})")
    if len(sub_categories) != 5:
        logger.warning(
            f"Expected 5 sub_categories but found {len(sub_categories)}. "
            "Proceeding anyway."
        )

    cv_cache = {"y": [], "pred": [], "wt": []}
    test_outputs: List[pd.DataFrame] = []
    pair_scores: Dict = {}
    feature_importance_all: Dict = {}

    for sc in sub_categories:
        logger.info("\n" + "=" * 70)
        logger.info(f"SUB-CATEGORY = '{sc}'")
        logger.info("=" * 70)

        logger.info(f"  Loading train data for sub_category='{sc}'...")
        sc_train_full = pd.read_parquet(TRAIN_PATH)
        sc_train_full = sc_train_full[sc_train_full["sub_category"] == sc].copy()
        logger.info(f"  Sub-category '{sc}' train rows: {len(sc_train_full):,}")

        has_test = False
        sc_test_full: Optional[pd.DataFrame] = None
        if TEST_PATH.exists():
            sc_test_full = pd.read_parquet(TEST_PATH)
            sc_test_full = sc_test_full[sc_test_full["sub_category"] == sc].copy()
            has_test = True
            logger.info(f"  Sub-category '{sc}' test rows : {len(sc_test_full):,}")

        for hz in HORIZONS:
            logger.info(f"\n  ── sub_category='{sc}' | Horizon = {hz} ──")

            hz_train = sc_train_full[sc_train_full["horizon"] == hz].copy()
            logger.info(f"    Total rows (train+val): {len(hz_train):,}")

            if len(hz_train) == 0:
                logger.warning(f"    No data for sc='{sc}', hz={hz}. Skipping.")
                continue

            hz_train = build_context_features(hz_train, train_stats, hz, logger)

            hz_test: Optional[pd.DataFrame] = None
            if has_test and sc_test_full is not None:
                hz_test = sc_test_full[sc_test_full["horizon"] == hz].copy()
                if len(hz_test) > 0:
                    hz_test = build_context_features(hz_test, train_stats, hz, logger)
                    logger.info(f"    Test rows: {len(hz_test):,}")

            exclude_cols = {
                "id", "code", "sub_code", "sub_category",
                "horizon", "ts_index", "weight", "y_target",
            }
            feature_cols = [c for c in hz_train.columns if c not in exclude_cols]
            logger.info(f"    Feature columns: {len(feature_cols)}")

            fit_mask = hz_train["ts_index"] <= VAL_THRESHOLD
            val_mask = hz_train["ts_index"] > VAL_THRESHOLD

            X_fit = hz_train.loc[fit_mask, feature_cols]
            y_fit = hz_train.loc[fit_mask, "y_target"]
            w_fit = hz_train.loc[fit_mask, "weight"]

            X_hold = hz_train.loc[val_mask, feature_cols]
            y_hold = hz_train.loc[val_mask, "y_target"]
            w_hold = hz_train.loc[val_mask, "weight"]

            logger.info(f"    Train: {len(X_fit):,} | Val: {len(X_hold):,}")

            if len(X_fit) == 0 or len(X_hold) == 0:
                logger.warning("    Insufficient train/val rows. Skipping.")
                del hz_train
                gc.collect()
                continue

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

            imp_df = pd.DataFrame(
                {"feature": feature_cols, "importance": fi_accum}
            ).sort_values("importance", ascending=False)
            feature_importance_all[(sc, hz)] = imp_df

            zeroed = imp_df[imp_df["importance"] == 0]
            logger.info(f"    L1 zeroed features : {len(zeroed)}")
            logger.info(f"    Top-10 features    : {imp_df.head(10)['feature'].tolist()}")

            cv_cache["y"].extend(y_hold.tolist())
            cv_cache["pred"].extend(val_pred.tolist())
            cv_cache["wt"].extend(w_hold.tolist())

            score = weighted_rmse_score(y_hold, val_pred, w_hold)
            pair_scores[(sc, hz)] = score
            logger.info(f"    Score (sc='{sc}', hz={hz}): {score:.5f}")

            if tst_pred is not None and hz_test is not None and len(hz_test) > 0:
                test_outputs.append(
                    pd.DataFrame({"id": hz_test["id"].values, "prediction": tst_pred})
                )

            del hz_train
            if hz_test is not None:
                del hz_test
            gc.collect()

        del sc_train_full
        if sc_test_full is not None:
            del sc_test_full
        gc.collect()

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
