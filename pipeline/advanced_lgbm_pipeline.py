#!/usr/bin/env python3
"""
Advanced LightGBM Horizon Pipeline

Features:
1. Target encoding for categorical features (sub_category, sub_code)
2. Lag features (1, 3, 10 steps)
3. Rolling statistics (mean, std) with windows 5, 10
4. EWM features (span=5)
5. Difference and ratio features
6. Cross-sectional rank features
7. Multi-seed ensemble (5 seeds averaged)
8. Sample weights during training
9. Time-based split (val_threshold=3500)
"""

import logging
import os
import sys
import gc
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from typing import Tuple, List, Dict, Optional

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "combined"
TRAIN_PATH = DATA_RAW_DIR / "train.parquet"
TEST_PATH = DATA_RAW_DIR / "test.parquet"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 10, 25]
VAL_THRESHOLD = 3500  # Time-based split threshold


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_file: str = "advanced_lgbm_pipeline.log") -> logging.Logger:
    """Configure logging to console and file."""
    logger = logging.getLogger("AdvancedLGBM")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# WEIGHTED RMSE SCORE (Competition Metric)
# ============================================================================
def weighted_rmse_score(y_target, y_pred, w) -> float:
    """
    Calculate weighted RMSE score (competition metric).
    
    Score = sqrt(1 - clipped(sum(w*(y-pred)^2) / sum(w*y^2)))
    """
    y_target, y_pred, w = np.array(y_target), np.array(y_pred), np.array(w)
    denom = np.sum(w * (y_target ** 2))
    if denom <= 0:
        return 0.0
    numerator = np.sum(w * ((y_target - y_pred) ** 2))
    ratio = numerator / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))


# ============================================================================
# COMPUTE TARGET ENCODING STATS
# ============================================================================
def compute_train_stats(train_path: Path, val_threshold: int, logger: logging.Logger) -> Dict:
    """
    Compute target encoding statistics from training data only.
    
    Uses only data up to val_threshold to avoid leakage.
    """
    logger.info("Computing target encoding statistics...")
    
    temp = pd.read_parquet(
        train_path,
        columns=['sub_category', 'sub_code', 'y_target', 'ts_index']
    )
    
    train_only = temp[temp.ts_index <= val_threshold]
    
    train_stats = {
        'sub_category': train_only.groupby('sub_category')['y_target'].mean().to_dict(),
        'sub_code': train_only.groupby('sub_code')['y_target'].mean().to_dict(),
        'global_mean': train_only['y_target'].mean()
    }
    
    logger.info(f"  sub_category encodings: {len(train_stats['sub_category'])}")
    logger.info(f"  sub_code encodings: {len(train_stats['sub_code'])}")
    logger.info(f"  global_mean: {train_stats['global_mean']:.6f}")
    
    del temp, train_only
    gc.collect()
    
    return train_stats


# ============================================================================
# CONTEXT FEATURE ENGINEERING
# ============================================================================
def build_context_features(data: pd.DataFrame, enc_stats: Dict = None, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Build advanced context features:
    - Target encodings for categorical columns
    - Lag features
    - Rolling mean/std
    - EWM features
    - Difference and ratio features
    - Cross-sectional rank features
    """
    x = data.copy()
    group_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    top_features = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'feature_s']
    
    # Encoded categorical signals (target encoding)
    if enc_stats is not None:
        for c in ['sub_category', 'sub_code']:
            x[c + '_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # Interaction features
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['r_al_am'] = x['feature_al'] / (x['feature_am'] + 1e-7)
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']
    
    # Lag, rolling, and EWM features for top features
    for col in top_features:
        if col not in x.columns:
            continue
        
        # Lag features
        for lag in [1, 3, 10]:
            x[f'{col}_lag{lag}'] = x.groupby(group_cols)[col].shift(lag).astype(np.float32)
        
        # Diff features
        x[f'{col}_diff1'] = x.groupby(group_cols)[col].diff(1).astype(np.float32)
        
        # Rolling mean and std
        for window in [5, 10]:
            x[f'{col}_roll{window}'] = x.groupby(group_cols)[col].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            ).astype(np.float32)
            
            x[f'{col}_rollstd{window}'] = x.groupby(group_cols)[col].transform(
                lambda s: s.rolling(window, min_periods=1).std()
            ).astype(np.float32)
        
        # EWM features
        x[f'{col}_ewm5'] = x.groupby(group_cols)[col].transform(
            lambda s: s.ewm(span=5, adjust=False).mean()
        ).astype(np.float32)
    
    # Temporal signal
    x['t_cycle'] = np.sin(2 * np.pi * x['ts_index'] / 100)
    
    # Cross-sectional rank features
    for col in ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am']:
        if col in x.columns:
            x[col + '_rk'] = x.groupby('ts_index')[col].rank(pct=True).astype(np.float32)
    
    return x


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main pipeline execution."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("ADVANCED LIGHTGBM HORIZON PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Validation threshold: ts_index > {VAL_THRESHOLD}")
    
    # LightGBM configuration
    lgb_cfg = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.02,
        'n_estimators': 6000,
        'num_leaves': 96,
        'min_child_samples': 150,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.2,
        'lambda_l2': 15.0,
        'verbosity': -1
    }
    
    logger.info(f"LightGBM config: lr={lgb_cfg['learning_rate']}, leaves={lgb_cfg['num_leaves']}")
    
    # Compute target encoding stats
    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)
    
    # Cross-validation cache for combined score
    cv_cache = {'y': [], 'pred': [], 'wt': []}
    test_outputs = []
    horizon_scores = {}
    
    # Ensemble seeds
    seeds = [42, 2024, 7, 11, 999]
    logger.info(f"Ensemble seeds: {seeds}")
    
    # Process each horizon
    for hz in HORIZONS:
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING HORIZON = {hz}")
        logger.info("=" * 60)
        
        # Load and filter data for this horizon
        logger.info(f"Loading horizon {hz} data...")
        tr_df = pd.read_parquet(TRAIN_PATH)
        tr_df = tr_df[tr_df['horizon'] == hz].copy()
        logger.info(f"Train data: {len(tr_df):,} rows")
        
        # Build context features
        logger.info("Building context features...")
        tr_df = build_context_features(tr_df, train_stats, logger)
        
        # Check if test file exists
        if TEST_PATH.exists():
            te_df = pd.read_parquet(TEST_PATH)
            te_df = te_df[te_df['horizon'] == hz].copy()
            te_df = build_context_features(te_df, train_stats, logger)
            has_test = True
            logger.info(f"Test data: {len(te_df):,} rows")
        else:
            has_test = False
            logger.info("No test file found, skipping test predictions")
        
        # Define feature columns
        exclude_cols = {
            'id', 'code', 'sub_code', 'sub_category',
            'horizon', 'ts_index', 'weight', 'y_target'
        }
        feature_cols = [c for c in tr_df.columns if c not in exclude_cols]
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        # Time-based split
        fit_mask = tr_df.ts_index <= VAL_THRESHOLD
        val_mask = tr_df.ts_index > VAL_THRESHOLD
        
        X_fit = tr_df.loc[fit_mask, feature_cols]
        y_fit = tr_df.loc[fit_mask, 'y_target']
        w_fit = tr_df.loc[fit_mask, 'weight']
        
        X_hold = tr_df.loc[val_mask, feature_cols]
        y_hold = tr_df.loc[val_mask, 'y_target']
        w_hold = tr_df.loc[val_mask, 'weight']
        
        logger.info(f"Train: {len(X_fit):,} rows (ts_index <= {VAL_THRESHOLD})")
        logger.info(f"Val:   {len(X_hold):,} rows (ts_index > {VAL_THRESHOLD})")
        
        # Multi-seed ensemble
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
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )
            
            val_pred += mdl.predict(X_hold) / len(seeds)
            if has_test:
                tst_pred += mdl.predict(te_df[feature_cols]) / len(seeds)
            
            logger.info(f"    Best iteration: {mdl.best_iteration_}")
        
        # Store CV results
        cv_cache['y'].extend(y_hold.tolist())
        cv_cache['pred'].extend(val_pred.tolist())
        cv_cache['wt'].extend(w_hold.tolist())
        
        # Horizon score
        hz_score = weighted_rmse_score(y_hold, val_pred, w_hold)
        horizon_scores[hz] = hz_score
        logger.info(f"\nHorizon {hz} Score: {hz_score:.5f}")
        
        # Store test predictions
        if has_test:
            test_outputs.append(
                pd.DataFrame({'id': te_df['id'], 'prediction': tst_pred})
            )
        
        # Cleanup
        del tr_df
        if has_test:
            del te_df
        gc.collect()
    
    # Final combined score
    final_score = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt'])
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"FINAL AGGREGATE SCORE: {final_score:.6f}")
    logger.info("\nPer-Horizon Scores:")
    for hz, score in horizon_scores.items():
        logger.info(f"  Horizon {hz:2d}: {score:.5f}")
    
    # Save results
    results_path = OUTPUTS_DIR / "advanced_lgbm_results.txt"
    with open(results_path, 'w') as f:
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
    
    # Save submission if test data exists
    if test_outputs:
        submission = pd.concat(test_outputs)
        submission_path = OUTPUTS_DIR / "submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to: {submission_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'final_score': final_score,
        'horizon_scores': horizon_scores
    }


if __name__ == "__main__":
    main()
