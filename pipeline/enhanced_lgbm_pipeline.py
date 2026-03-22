#!/usr/bin/env python3
"""
Enhanced LightGBM Horizon Pipeline with Discovered Interaction Features

Features from gplearn, PySR, and tsfresh discoveries:
1. GPlearn Combined: feature_l × feature_ca × feature_t
2. GPlearn H1: feature_cd / feature_c × feature_q
3. GPlearn H3: 3 × feature_bz
4. GPlearn H10: complex interactions with feature_ah, feature_w, feature_bv, feature_bs, feature_x
5. GPlearn H25: feature_b × feature_bz, feature_bx × feature_bz, feature_n × feature_bz
6. PySR Combined: feature_bz × feature_t², feature_v × feature_u
7. PySR H1: 1 / (const - feature_aw), feature_d × feature_s
8. PySR H3: feature_bz × feature_f, feature_bz / (const - feature_f)
9. PySR H10: feature_z × feature_bz², feature_bz × |feature_bz|
10. PySR H25: feature_s × (feature_bz + const), feature_e × feature_bz

Plus L1 regularization for feature selection.
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
VAL_THRESHOLD = 3500


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_file: str = "enhanced_lgbm_pipeline.log") -> logging.Logger:
    """Configure logging to console and file."""
    logger = logging.getLogger("EnhancedLGBM")
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
    """Calculate weighted RMSE score (competition metric)."""
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
    """Compute target encoding statistics from training data only."""
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
# DISCOVERED INTERACTION FEATURES (from gplearn, PySR, tsfresh)
# ============================================================================
def build_discovered_features(x: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Build interaction features discovered by symbolic regression.
    
    Sources:
    - GPlearn combined: mul(feature_l, mul(feature_ca, feature_t))
    - GPlearn H1: mul(div(feature_cd, feature_c), feature_q)
    - GPlearn H3: add(feature_bz, add(feature_bz, feature_bz)) = 3*feature_bz
    - GPlearn H10: complex with feature_ah, feature_w, feature_bv, feature_bs, feature_x, feature_aa
    - GPlearn H25: feature_b × feature_bz, feature_bx × feature_bz, feature_n × feature_bz
    - PySR combined: feature_bz*feature_t², abs(feature_v*feature_u)
    - PySR H1: feature_d × feature_s
    - PySR H3: feature_bz × feature_f, feature_bz / (0.88 - feature_f)
    - PySR H10: feature_z × feature_bz², feature_bz × |feature_bz|
    - PySR H25: feature_s × (feature_bz + 1.48), feature_e × feature_bz
    """
    eps = 1e-7
    
    # ======== GPLEARN COMBINED DISCOVERIES ========
    x['gp_l_ca_t'] = x['feature_l'] * x['feature_ca'] * x['feature_t']
    x['gp_ca_t'] = x['feature_ca'] * x['feature_t']
    
    # ======== PYSR COMBINED DISCOVERIES ========
    x['psr_bz_t2'] = x['feature_bz'] * (x['feature_t'] ** 2)
    x['psr_bz_div_bp_t2'] = x['feature_bz'] / (x['feature_bp'] / (x['feature_t'] ** 2 + eps) + eps)
    x['psr_vu_abs'] = np.abs(x['feature_v'] * x['feature_u'])
    x['psr_bz_t'] = x['feature_bz'] * x['feature_t']
    x['psr_neg_s'] = -x['feature_s']
    x['psr_u_scaled'] = x['feature_u'] * -0.07338854
    
    # ======== HORIZON 1 DISCOVERIES ========
    # GPlearn: mul(div(feature_cd, feature_c), feature_q)
    x['h1_cd_div_c_q'] = (x['feature_cd'] / (x['feature_c'] + eps)) * x['feature_q']
    # PySR: -7.47 / (2.74 - feature_aw)
    x['h1_aw_inv'] = -7.465855 / (2.7417014 - x['feature_aw'] + eps)
    # PySR: feature_d × feature_s
    x['h1_d_s'] = x['feature_d'] * x['feature_s']
    x['h1_abs_bz'] = np.abs(x['feature_bz'])
    
    # ======== HORIZON 3 DISCOVERIES ========
    # GPlearn: 3 × feature_bz
    x['h3_bz_triple'] = 3 * x['feature_bz']
    # PySR: feature_bz + feature_bz / (0.88 - feature_f)
    x['h3_bz_f_ratio'] = x['feature_bz'] + x['feature_bz'] / (0.87934214 - x['feature_f'] + eps)
    # PySR: (feature_f - 7.83) × feature_bz
    x['h3_f_bz'] = (x['feature_f'] - 7.829302) * x['feature_bz']
    # PySR: (feature_bz × feature_f) - (feature_b × feature_bz)
    x['h3_bz_f_minus_b_bz'] = (x['feature_bz'] * x['feature_f']) - (x['feature_b'] * x['feature_bz'])
    
    # ======== HORIZON 10 DISCOVERIES ========
    # GPlearn: complex with feature_ah, feature_w, feature_bv, feature_bs, feature_x, feature_aa
    x['h10_ah_w_bv'] = x['feature_ah'] + x['feature_w'] * x['feature_bv']
    x['h10_bs_x_aa'] = (x['feature_bs'] + x['feature_x']) * x['feature_aa']
    x['h10_ar_ch'] = x['feature_ar'] - x['feature_ch']
    x['h10_complex'] = (x['h10_ah_w_bv'] + x['h10_bs_x_aa'] + x['h10_ar_ch']) * x['feature_bz']
    # PySR: feature_z × feature_bz²
    x['h10_z_bz2'] = x['feature_z'] * (x['feature_bz'] ** 2)
    # PySR: feature_bz × |feature_bz|
    x['h10_bz_abs_bz'] = x['feature_bz'] * np.abs(x['feature_bz'])
    # PySR: feature_bz × feature_s
    x['h10_bz_s'] = x['feature_bz'] * x['feature_s']
    # PySR: feature_bm × (-0.0021)
    x['h10_bm_scaled'] = x['feature_bm'] * -0.0021058018
    
    # ======== HORIZON 25 DISCOVERIES ========
    # GPlearn: feature_b × feature_bz
    x['h25_b_bz'] = x['feature_b'] * x['feature_bz']
    # GPlearn: feature_bx × feature_bz
    x['h25_bx_bz'] = x['feature_bx'] * x['feature_bz']
    # GPlearn: feature_n × feature_bz
    x['h25_n_bz'] = x['feature_n'] * x['feature_bz']
    # GPlearn: feature_o + feature_bz
    x['h25_o_bz'] = x['feature_o'] + x['feature_bz']
    # PySR: feature_s × (feature_bz + 1.48)
    x['h25_s_bz_shift'] = x['feature_s'] * (x['feature_bz'] + 1.4804125)
    # PySR: feature_e × feature_bz
    x['h25_e_bz'] = x['feature_e'] * x['feature_bz']
    # PySR: |feature_cd| × feature_bz
    x['h25_cd_abs_bz'] = np.abs(x['feature_cd']) * x['feature_bz']
    
    # ======== ADDITIONAL DERIVED FEATURES ========
    # Polynomial features for key discovered variables
    x['bz_squared'] = x['feature_bz'] ** 2
    x['bz_cubed'] = x['feature_bz'] ** 3
    x['s_squared'] = x['feature_s'] ** 2
    x['t_squared'] = x['feature_t'] ** 2
    
    # Ratios discovered to be important
    x['al_div_am'] = x['feature_al'] / (x['feature_am'] + eps)
    x['bz_div_s'] = x['feature_bz'] / (x['feature_s'] + eps)
    x['cd_div_bz'] = x['feature_cd'] / (x['feature_bz'] + eps)
    x['bz_div_bp'] = x['feature_bz'] / (x['feature_bp'] + eps)
    
    # Cross-horizon important interactions
    x['al_bz'] = x['feature_al'] * x['feature_bz']
    x['al_s'] = x['feature_al'] * x['feature_s']
    x['v_u'] = x['feature_v'] * x['feature_u']
    x['cg_by'] = x['feature_cg'] * x['feature_by']
    
    return x


# ============================================================================
# CONTEXT FEATURE ENGINEERING (from advanced pipeline)
# ============================================================================
def build_context_features(data: pd.DataFrame, enc_stats: Dict, horizon: int, logger: logging.Logger = None) -> pd.DataFrame:
    """Build context features with target encoding, lag features, and discovered interactions."""
    x = data.copy()
    group_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    top_features = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'feature_s', 'feature_bz']
    
    # ======== TARGET ENCODING ========
    for c in ['sub_category', 'sub_code']:
        x[c + '_enc'] = x[c].map(enc_stats[c]).fillna(enc_stats['global_mean'])
    
    # ======== BASIC INTERACTION FEATURES ========
    x['d_al_am'] = x['feature_al'] - x['feature_am']
    x['r_al_am'] = x['feature_al'] / (x['feature_am'] + 1e-7)
    x['d_cg_by'] = x['feature_cg'] - x['feature_by']
    
    # ======== LAG FEATURES ========
    for col in top_features:
        if col not in x.columns:
            continue
        for lag in [1, 3, 10]:
            x[f'{col}_lag{lag}'] = x.groupby(group_cols)[col].shift(lag).astype(np.float32)
        x[f'{col}_diff1'] = x.groupby(group_cols)[col].diff(1).astype(np.float32)
    
    # ======== ROLLING FEATURES ========
    for col in top_features:
        if col not in x.columns:
            continue
        for window in [5, 10]:
            x[f'{col}_roll{window}'] = x.groupby(group_cols)[col].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            ).astype(np.float32)
            x[f'{col}_rollstd{window}'] = x.groupby(group_cols)[col].transform(
                lambda s: s.rolling(window, min_periods=1).std()
            ).astype(np.float32)
        x[f'{col}_ewm5'] = x.groupby(group_cols)[col].transform(
            lambda s: s.ewm(span=5, adjust=False).mean()
        ).astype(np.float32)
    
    # ======== TEMPORAL SIGNAL ========
    x['t_cycle'] = np.sin(2 * np.pi * x['ts_index'] / 100)
    
    # ======== CROSS-SECTIONAL RANK FEATURES ========
    for col in ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am', 'feature_bz']:
        if col in x.columns:
            x[col + '_rk'] = x.groupby('ts_index')[col].rank(pct=True).astype(np.float32)
    
    # ======== DISCOVERED INTERACTION FEATURES ========
    x = build_discovered_features(x, horizon)
    
    return x


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main pipeline execution."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("ENHANCED LIGHTGBM PIPELINE WITH DISCOVERED FEATURES + L1")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Validation threshold: ts_index > {VAL_THRESHOLD}")
    
    # LightGBM configuration with L1 regularization
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
        'lambda_l1': 0.2,      # L1 regularization for feature selection
        'lambda_l2': 15.0,     # L2 regularization
        'verbosity': -1
    }
    
    logger.info(f"LightGBM config: lr={lgb_cfg['learning_rate']}, L1={lgb_cfg['lambda_l1']}, L2={lgb_cfg['lambda_l2']}")
    
    # Compute target encoding stats
    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)
    
    # Cross-validation cache
    cv_cache = {'y': [], 'pred': [], 'wt': []}
    test_outputs = []
    horizon_scores = {}
    feature_importance_all = {}
    
    # Ensemble seeds
    seeds = [42, 2024, 7, 11, 999]
    logger.info(f"Ensemble seeds: {seeds}")
    
    # Process each horizon
    for hz in HORIZONS:
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING HORIZON = {hz}")
        logger.info("=" * 60)
        
        # Load and filter data
        logger.info(f"Loading horizon {hz} data...")
        tr_df = pd.read_parquet(TRAIN_PATH)
        tr_df = tr_df[tr_df['horizon'] == hz].copy()
        logger.info(f"Train data: {len(tr_df):,} rows")
        
        # Build features
        logger.info("Building context and discovered features...")
        tr_df = build_context_features(tr_df, train_stats, hz, logger)
        
        # Load test data if exists
        if TEST_PATH.exists():
            te_df = pd.read_parquet(TEST_PATH)
            te_df = te_df[te_df['horizon'] == hz].copy()
            te_df = build_context_features(te_df, train_stats, hz, logger)
            has_test = True
            logger.info(f"Test data: {len(te_df):,} rows")
        else:
            has_test = False
        
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
        
        logger.info(f"Train: {len(X_fit):,} rows, Val: {len(X_hold):,} rows")
        
        # Multi-seed ensemble
        val_pred = np.zeros(len(y_hold))
        if has_test:
            tst_pred = np.zeros(len(te_df))
        
        feature_importance_hz = np.zeros(len(feature_cols))
        
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
            
            # Accumulate feature importance
            feature_importance_hz += mdl.feature_importances_ / len(seeds)
            
            logger.info(f"    Best iteration: {mdl.best_iteration_}")
        
        # Store feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance_hz
        }).sort_values('importance', ascending=False)
        feature_importance_all[hz] = importance_df
        
        # Identify L1-zeroed features
        zeroed_features = importance_df[importance_df['importance'] == 0]['feature'].tolist()
        logger.info(f"L1 zeroed features: {len(zeroed_features)}")
        logger.info(f"Top 10 features: {importance_df.head(10)['feature'].tolist()}")
        
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
    results_path = OUTPUTS_DIR / "enhanced_lgbm_results.txt"
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ENHANCED LIGHTGBM PIPELINE WITH DISCOVERED FEATURES + L1\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"FINAL AGGREGATE SCORE: {final_score:.6f}\n\n")
        f.write("Per-Horizon Scores:\n")
        for hz, score in horizon_scores.items():
            f.write(f"  Horizon {hz:2d}: {score:.5f}\n")
        f.write("\n")
        f.write(f"Validation threshold: ts_index > {VAL_THRESHOLD}\n")
        f.write(f"Ensemble seeds: {seeds}\n")
        f.write(f"Total features: {len(feature_cols)}\n")
        f.write(f"L1 regularization: {lgb_cfg['lambda_l1']}\n")
        f.write(f"L2 regularization: {lgb_cfg['lambda_l2']}\n\n")
        
        # Feature importance per horizon
        for hz in HORIZONS:
            imp_df = feature_importance_all[hz]
            f.write(f"\n{'='*40}\n")
            f.write(f"HORIZON {hz} - TOP 20 FEATURES\n")
            f.write(f"{'='*40}\n")
            for i, row in imp_df.head(20).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.2f}\n")
            zeroed = imp_df[imp_df['importance'] == 0]
            f.write(f"\nL1 zeroed features: {len(zeroed)}\n")
    
    logger.info(f"Results saved to: {results_path}")
    
    # Save submission
    if test_outputs:
        submission = pd.concat(test_outputs)
        submission_path = OUTPUTS_DIR / "enhanced_submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to: {submission_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    
    return {'final_score': final_score, 'horizon_scores': horizon_scores}


if __name__ == "__main__":
    main()
