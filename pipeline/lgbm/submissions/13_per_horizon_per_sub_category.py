#!/usr/bin/env python3
"""Per-horizon × per-sub-category LightGBM — entry script."""
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
    logger = setup_logging("PerHzPerSubCat", "per_horizon_per_sub_category.log")

    logger.info("=" * 70)
    logger.info("PER-HORIZON PER-SUB-CATEGORY LIGHTGBM PIPELINE")
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
            "Proceeding with discovered values."
        )

    cv_cache = {"y": [], "pred": [], "wt": []}
    test_outputs: List[pd.DataFrame] = []
    pair_scores: Dict = {}
    feature_importance_all: Dict = {}

    for hz in HORIZONS:
        logger.info("\n" + "=" * 70)
        logger.info(f"HORIZON = {hz}")
        logger.info("=" * 70)

        logger.info(f"  Loading horizon {hz} train data...")
        hz_train = pd.read_parquet(TRAIN_PATH)
        hz_train = hz_train[hz_train["horizon"] == hz].copy()
        logger.info(f"  Horizon {hz} train rows: {len(hz_train):,}")

        hz_train = build_context_features(hz_train, train_stats, hz, logger)

        has_test = False
        hz_test: Optional[pd.DataFrame] = None
        if TEST_PATH.exists():
            hz_test = pd.read_parquet(TEST_PATH)
            hz_test = hz_test[hz_test["horizon"] == hz].copy()
            hz_test = build_context_features(hz_test, train_stats, hz, logger)
            has_test = True
            logger.info(f"  Horizon {hz} test rows : {len(hz_test):,}")

        exclude_cols = {
            "id", "code", "sub_code", "sub_category",
            "horizon", "ts_index", "weight", "y_target",
        }
        feature_cols = [c for c in hz_train.columns if c not in exclude_cols]
        logger.info(f"  Feature columns: {len(feature_cols)}")

        for sc in sub_categories:
            logger.info(f"\n  ── Horizon {hz} | sub_category = '{sc}' ──")

            sc_train = hz_train[hz_train["sub_category"] == sc].copy()
            logger.info(f"    Total rows (train+val): {len(sc_train):,}")

            if len(sc_train) == 0:
                logger.warning(f"    No data for hz={hz}, sc={sc}. Skipping.")
                continue

            fit_mask = sc_train["ts_index"] <= VAL_THRESHOLD
            val_mask = sc_train["ts_index"] > VAL_THRESHOLD

            X_fit = sc_train.loc[fit_mask, feature_cols]
            y_fit = sc_train.loc[fit_mask, "y_target"]
            w_fit = sc_train.loc[fit_mask, "weight"]

            X_hold = sc_train.loc[val_mask, feature_cols]
            y_hold = sc_train.loc[val_mask, "y_target"]
            w_hold = sc_train.loc[val_mask, "weight"]

            logger.info(f"    Train: {len(X_fit):,} rows | Val: {len(X_hold):,} rows")

            if len(X_fit) == 0 or len(X_hold) == 0:
                logger.warning("    Insufficient train/val rows. Skipping.")
                continue

            sc_test: Optional[pd.DataFrame] = None
            if has_test and hz_test is not None:
                sc_test = hz_test[hz_test["sub_category"] == sc]

            val_pred = np.zeros(len(y_hold))
            tst_pred = (
                np.zeros(len(sc_test))
                if sc_test is not None and len(sc_test) > 0
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
                if tst_pred is not None and sc_test is not None and len(sc_test) > 0:
                    tst_pred += mdl.predict(sc_test[feature_cols]) / len(seeds)
                fi_accum += mdl.feature_importances_ / len(seeds)
                logger.info(f"      Best iteration: {mdl.best_iteration_}")

            imp_df = pd.DataFrame(
                {"feature": feature_cols, "importance": fi_accum}
            ).sort_values("importance", ascending=False)
            feature_importance_all[(hz, sc)] = imp_df

            zeroed = imp_df[imp_df["importance"] == 0]
            logger.info(f"    L1 zeroed features : {len(zeroed)}")
            logger.info(f"    Top-10 features    : {imp_df.head(10)['feature'].tolist()}")

            cv_cache["y"].extend(y_hold.tolist())
            cv_cache["pred"].extend(val_pred.tolist())
            cv_cache["wt"].extend(w_hold.tolist())

            score = weighted_rmse_score(y_hold, val_pred, w_hold)
            pair_scores[(hz, sc)] = score
            logger.info(f"    Score (hz={hz}, sc='{sc}'): {score:.5f}")

            if tst_pred is not None and sc_test is not None and len(sc_test) > 0:
                test_outputs.append(
                    pd.DataFrame({"id": sc_test["id"].values, "prediction": tst_pred})
                )

            del sc_train
            gc.collect()

        del hz_train
        if hz_test is not None:
            del hz_test
        gc.collect()

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

        for (hz, sc), imp_df in feature_importance_all.items():
            f.write(f"\n{'='*40}\n")
            f.write(f"HORIZON {hz} | sub_category='{sc}' — TOP 20 FEATURES\n")
            f.write(f"{'='*40}\n")
            for _, row in imp_df.head(20).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.2f}\n")
            zeroed = imp_df[imp_df["importance"] == 0]
            f.write(f"\nL1 zeroed features: {len(zeroed)}\n")

    logger.info(f"Results saved to: {results_path}")

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
