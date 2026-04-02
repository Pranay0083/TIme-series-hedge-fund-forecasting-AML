#!/usr/bin/env python3
"""Per-horizon IC analysis + LightGBM — entry script."""
import importlib.util
import os
import sys
from datetime import datetime

import gc
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
mod_07 = import_local_module("07_ic_analysis")

TRAIN_PATH = mod_01.TRAIN_PATH
TEST_PATH = mod_01.TEST_PATH
OUTPUTS_DIR = mod_01.OUTPUTS_DIR
HORIZONS = mod_01.HORIZONS
VAL_THRESHOLD = mod_01.VAL_THRESHOLD
setup_logging = mod_02.setup_logging
weighted_rmse_score = mod_03.weighted_rmse_score
compute_train_stats = mod_04.compute_train_stats
build_context_features = mod_06.build_context_features
get_feature_groups = mod_07.get_feature_groups
analyze_feature_group_ic = mod_07.analyze_feature_group_ic
train_horizon_model = mod_07.train_horizon_model


def main():
    logger = setup_logging("PerHorizonIC", "perhorizon_ic_pipeline.log")

    logger.info("=" * 70)
    logger.info("PER-HORIZON IC ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Validation threshold: ts_index > {VAL_THRESHOLD}")

    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)
    cv_cache = {"y": [], "pred": [], "wt": []}
    test_outputs = []
    horizon_scores = {}
    horizon_ic_analysis = {}
    feature_importance_all = {}
    seeds = [42, 2024, 7, 11, 999]
    logger.info(f"Ensemble seeds: {seeds}")

    for hz in HORIZONS:
        logger.info("\n" + "=" * 70)
        logger.info(f"HORIZON {hz} - INDEPENDENT MODEL + IC ANALYSIS")
        logger.info("=" * 70)

        logger.info(f"Loading horizon {hz} data...")
        tr_df = pd.read_parquet(TRAIN_PATH)
        tr_df = tr_df[tr_df["horizon"] == hz].copy()
        logger.info(f"Train data: {len(tr_df):,} rows")

        logger.info("Building context and discovered features...")
        tr_df = build_context_features(tr_df, train_stats, hz, logger)

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

        logger.info(f"Train: {len(X_fit):,} rows, Val: {len(X_hold):,} rows")

        logger.info("\n--- IC/RankIC Analysis per Feature Group ---")
        feature_groups = get_feature_groups(feature_cols)
        ic_results = {}
        for group_name, group_cols in feature_groups.items():
            if not group_cols:
                continue
            result = analyze_feature_group_ic(
                tr_df.loc[val_mask],
                group_cols,
                "y_target",
                group_name,
                logger,
                weight_col="weight",
            )
            ic_results[group_name] = result
            logger.info(
                f"  {group_name:25s}: n={result['n_features']:3d}, "
                f"IC={result['mean_ic']:.4f}, RankIC={result['mean_rank_ic']:.4f}, "
                f"MaxIC={result['max_ic']:.4f} ({result['max_ic_feature']})"
            )

        horizon_ic_analysis[hz] = ic_results

        logger.info("\n--- Training Independent Horizon Model ---")
        val_pred, model_info, trained_models = train_horizon_model(
            X_fit, y_fit, w_fit, X_hold, y_hold, w_hold, hz, logger, seeds
        )
        feature_importance_all[hz] = model_info["importance"]

        cv_cache["y"].extend(y_hold.tolist())
        cv_cache["pred"].extend(val_pred.tolist())
        cv_cache["wt"].extend(w_hold.tolist())

        hz_score = weighted_rmse_score(y_hold, val_pred, w_hold)
        horizon_scores[hz] = hz_score
        logger.info(f"\nHorizon {hz} Score: {hz_score:.5f}")

        if TEST_PATH.exists():
            logger.info("\n--- Generating Test Predictions ---")
            te_df = pd.read_parquet(TEST_PATH)
            te_df = te_df[te_df["horizon"] == hz].copy()
            te_df = build_context_features(te_df, train_stats, hz, logger)
            logger.info(f"Test data: {len(te_df):,} rows")

            tst_pred = np.zeros(len(te_df))
            for mdl in trained_models:
                tst_pred += mdl.predict(te_df[feature_cols]) / len(trained_models)

            test_outputs.append(pd.DataFrame({"id": te_df["id"], "prediction": tst_pred}))
            logger.info(f"Test predictions generated for horizon {hz}")
            del te_df

        del tr_df, trained_models
        gc.collect()

    final_score = weighted_rmse_score(cv_cache["y"], cv_cache["pred"], cv_cache["wt"])

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"FINAL AGGREGATE SCORE: {final_score:.6f}")
    logger.info("\nPer-Horizon Scores:")
    for hz, score in horizon_scores.items():
        logger.info(f"  Horizon {hz:2d}: {score:.5f}")

    results_path = OUTPUTS_DIR / "perhorizon_ic_results.txt"
    with open(results_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PER-HORIZON IC ANALYSIS PIPELINE RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"FINAL AGGREGATE SCORE: {final_score:.6f}\n\n")
        f.write("Per-Horizon Scores:\n")
        for hz, score in horizon_scores.items():
            f.write(f"  Horizon {hz:2d}: {score:.5f}\n")

        for hz in HORIZONS:
            f.write(f"\n{'='*70}\n")
            f.write(f"HORIZON {hz} - IC/RANKIC ANALYSIS BY FEATURE GROUP\n")
            f.write(f"{'='*70}\n\n")

            ic_results = horizon_ic_analysis[hz]
            sorted_groups = sorted(
                ic_results.items(), key=lambda x: x[1]["mean_ic"], reverse=True
            )

            f.write(f"{'Group':<30} {'N':>5} {'MeanIC':>10} {'MeanRankIC':>12} {'MaxIC':>10} {'MaxFeature'}\n")
            f.write("-" * 100 + "\n")

            for group_name, result in sorted_groups:
                f.write(
                    f"{group_name:<30} {result['n_features']:>5} "
                    f"{result['mean_ic']:>10.4f} {result['mean_rank_ic']:>12.4f} "
                    f"{result['max_ic']:>10.4f} {result['max_ic_feature']}\n"
                )

            f.write("\n--- Top 15 Features by Importance ---\n")
            imp_df = feature_importance_all[hz]
            for _, row in imp_df.head(15).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.2f}\n")

    logger.info(f"Results saved to: {results_path}")

    heatmap_data = []
    for hz in HORIZONS:
        for group_name, result in horizon_ic_analysis[hz].items():
            heatmap_data.append({
                "horizon": hz,
                "group": group_name,
                "mean_ic": result["mean_ic"],
                "mean_rank_ic": result["mean_rank_ic"],
                "max_ic": result["max_ic"],
                "n_features": result["n_features"],
            })

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_path = OUTPUTS_DIR / "ic_heatmap_data.csv"
    heatmap_df.to_csv(heatmap_path, index=False)
    logger.info(f"IC heatmap data saved to: {heatmap_path}")

    group_avg = heatmap_df.groupby("group").agg({
        "mean_ic": "mean",
        "mean_rank_ic": "mean",
        "n_features": "mean",
    }).reset_index().sort_values("mean_ic", ascending=False)

    ranking_path = OUTPUTS_DIR / "feature_group_ranking.csv"
    group_avg.to_csv(ranking_path, index=False)
    logger.info(f"Feature group ranking saved to: {ranking_path}")

    if test_outputs:
        submission = pd.concat(test_outputs)
        submission_path = OUTPUTS_DIR / "perhorizon_ic_submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to: {submission_path}")
        logger.info(f"Submission rows: {len(submission):,}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)

    return {
        "final_score": final_score,
        "horizon_scores": horizon_scores,
        "ic_analysis": horizon_ic_analysis,
    }


if __name__ == "__main__":
    main()
