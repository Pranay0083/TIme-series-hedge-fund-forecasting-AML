#!/usr/bin/env python3
"""
Per-Horizon IC Analysis Pipeline

Features:
1. Per-horizon model separation - Train independent models per horizon
2. IC / RankIC per feature group - Analyze predictive power by group

Feature Groups:
- Category encodings: sub_category_enc, sub_code_enc
- Volatility stats: rollstd, diff1 features
- Lag features: lag1, lag3, lag10
- Rolling features: roll5, roll10, ewm5
- Rank features: cross-sectional ranks
- Discovered interactions: GPlearn/PySR features
- Raw signals: original feature_* columns
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
from scipy import stats
from typing import Tuple, List, Dict, Optional, Callable

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
# FEATURE GROUP DEFINITIONS
# ============================================================================
def get_feature_groups(all_columns: List[str]) -> Dict[str, List[str]]:
    """Define feature groups for IC analysis."""
    groups = {
        'category_encodings': [c for c in all_columns if c.endswith('_enc')],
        'volatility_stats': [c for c in all_columns if 'rollstd' in c or 'diff1' in c],
        'lag_features': [c for c in all_columns if '_lag' in c],
        'rolling_features': [c for c in all_columns if '_roll' in c and 'std' not in c],
        'ewm_features': [c for c in all_columns if '_ewm' in c],
        'rank_features': [c for c in all_columns if '_rk' in c],
        'discovered_gp_psr': [c for c in all_columns if c.startswith('gp_') or c.startswith('psr_')],
        'discovered_horizon': [c for c in all_columns if c.startswith(('h1_', 'h3_', 'h10_', 'h25_'))],
        'polynomial': [c for c in all_columns if c in ['bz_squared', 'bz_cubed', 's_squared', 't_squared']],
        'ratios': [c for c in all_columns if c in ['al_div_am', 'bz_div_s', 'cd_div_bz', 'bz_div_bp']],
        'raw_signals': [c for c in all_columns if c.startswith('feature_')],
    }
    # Filter empty groups
    return {k: v for k, v in groups.items() if len(v) > 0}


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_file: str = "perhorizon_ic_pipeline.log") -> logging.Logger:
    """Configure logging to console and file."""
    logger = logging.getLogger("PerHorizonIC")
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
# IC/RANKIC COMPUTATION
# ============================================================================
def compute_ic(feature: np.ndarray, target: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Compute Information Coefficient (Pearson correlation)."""
    mask = ~(np.isnan(feature) | np.isnan(target))
    if mask.sum() < 10:
        return np.nan
    f, t = feature[mask], target[mask]
    if np.std(f) < 1e-10 or np.std(t) < 1e-10:
        return 0.0
    return np.corrcoef(f, t)[0, 1]


def compute_rank_ic(feature: np.ndarray, target: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Compute Rank IC (Spearman correlation)."""
    mask = ~(np.isnan(feature) | np.isnan(target))
    if mask.sum() < 10:
        return np.nan
    f, t = feature[mask], target[mask]
    corr, _ = stats.spearmanr(f, t)
    return corr


def compute_weighted_ic(feature: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted Information Coefficient."""
    mask = ~(np.isnan(feature) | np.isnan(target) | np.isnan(weights))
    if mask.sum() < 10:
        return np.nan
    f, t, w = feature[mask], target[mask], weights[mask]
    
    # Weighted means
    w_sum = w.sum()
    f_mean = (w * f).sum() / w_sum
    t_mean = (w * t).sum() / w_sum
    
    # Weighted covariance and standard deviations
    cov = (w * (f - f_mean) * (t - t_mean)).sum() / w_sum
    f_std = np.sqrt((w * (f - f_mean) ** 2).sum() / w_sum)
    t_std = np.sqrt((w * (t - t_mean) ** 2).sum() / w_sum)
    
    if f_std < 1e-10 or t_std < 1e-10:
        return 0.0
    return cov / (f_std * t_std)


def analyze_feature_group_ic(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    weight_col: str,
    group_name: str,
    logger: logging.Logger
) -> Dict:
    """Analyze IC metrics for a feature group."""
    target = df[target_col].values
    weights = df[weight_col].values
    
    results = {
        'group': group_name,
        'n_features': len(feature_cols),
        'features': {},
        'mean_ic': 0.0,
        'mean_rank_ic': 0.0,
        'mean_weighted_ic': 0.0,
        'max_ic': 0.0,
        'max_ic_feature': None,
    }
    
    ics, rank_ics, weighted_ics = [], [], []
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        feature = df[col].values
        
        ic = compute_ic(feature, target)
        rank_ic = compute_rank_ic(feature, target)
        w_ic = compute_weighted_ic(feature, target, weights)
        
        results['features'][col] = {
            'ic': ic,
            'rank_ic': rank_ic,
            'weighted_ic': w_ic
        }
        
        if not np.isnan(ic):
            ics.append(abs(ic))
        if not np.isnan(rank_ic):
            rank_ics.append(abs(rank_ic))
        if not np.isnan(w_ic):
            weighted_ics.append(abs(w_ic))
    
    if ics:
        results['mean_ic'] = np.mean(ics)
        results['max_ic'] = max(ics)
        max_idx = ics.index(max(ics))
        results['max_ic_feature'] = list(results['features'].keys())[max_idx]
    if rank_ics:
        results['mean_rank_ic'] = np.mean(rank_ics)
    if weighted_ics:
        results['mean_weighted_ic'] = np.mean(weighted_ics)
    
    return results


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
    """Build interaction features discovered by symbolic regression."""
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
    x['h1_cd_div_c_q'] = (x['feature_cd'] / (x['feature_c'] + eps)) * x['feature_q']
    x['h1_aw_inv'] = -7.465855 / (2.7417014 - x['feature_aw'] + eps)
    x['h1_d_s'] = x['feature_d'] * x['feature_s']
    x['h1_abs_bz'] = np.abs(x['feature_bz'])
    
    # ======== HORIZON 3 DISCOVERIES ========
    x['h3_bz_triple'] = 3 * x['feature_bz']
    x['h3_bz_f_ratio'] = x['feature_bz'] + x['feature_bz'] / (0.87934214 - x['feature_f'] + eps)
    x['h3_f_bz'] = (x['feature_f'] - 7.829302) * x['feature_bz']
    x['h3_bz_f_minus_b_bz'] = (x['feature_bz'] * x['feature_f']) - (x['feature_b'] * x['feature_bz'])
    
    # ======== HORIZON 10 DISCOVERIES ========
    x['h10_ah_w_bv'] = x['feature_ah'] + x['feature_w'] * x['feature_bv']
    x['h10_bs_x_aa'] = (x['feature_bs'] + x['feature_x']) * x['feature_aa']
    x['h10_ar_ch'] = x['feature_ar'] - x['feature_ch']
    x['h10_complex'] = (x['h10_ah_w_bv'] + x['h10_bs_x_aa'] + x['h10_ar_ch']) * x['feature_bz']
    x['h10_z_bz2'] = x['feature_z'] * (x['feature_bz'] ** 2)
    x['h10_bz_abs_bz'] = x['feature_bz'] * np.abs(x['feature_bz'])
    x['h10_bz_s'] = x['feature_bz'] * x['feature_s']
    x['h10_bm_scaled'] = x['feature_bm'] * -0.0021058018
    
    # ======== HORIZON 25 DISCOVERIES ========
    x['h25_b_bz'] = x['feature_b'] * x['feature_bz']
    x['h25_bx_bz'] = x['feature_bx'] * x['feature_bz']
    x['h25_n_bz'] = x['feature_n'] * x['feature_bz']
    x['h25_o_bz'] = x['feature_o'] + x['feature_bz']
    x['h25_s_bz_shift'] = x['feature_s'] * (x['feature_bz'] + 1.4804125)
    x['h25_e_bz'] = x['feature_e'] * x['feature_bz']
    x['h25_cd_abs_bz'] = np.abs(x['feature_cd']) * x['feature_bz']
    
    # ======== POLYNOMIAL FEATURES ========
    x['bz_squared'] = x['feature_bz'] ** 2
    x['bz_cubed'] = x['feature_bz'] ** 3
    x['s_squared'] = x['feature_s'] ** 2
    x['t_squared'] = x['feature_t'] ** 2
    
    # ======== RATIO FEATURES ========
    x['al_div_am'] = x['feature_al'] / (x['feature_am'] + eps)
    x['bz_div_s'] = x['feature_bz'] / (x['feature_s'] + eps)
    x['cd_div_bz'] = x['feature_cd'] / (x['feature_bz'] + eps)
    x['bz_div_bp'] = x['feature_bz'] / (x['feature_bp'] + eps)
    
    # ======== CROSS-HORIZON INTERACTIONS ========
    x['al_bz'] = x['feature_al'] * x['feature_bz']
    x['al_s'] = x['feature_al'] * x['feature_s']
    x['v_u'] = x['feature_v'] * x['feature_u']
    x['cg_by'] = x['feature_cg'] * x['feature_by']
    
    return x


# ============================================================================
# CONTEXT FEATURE ENGINEERING
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
# PER-HORIZON MODEL TRAINING
# ============================================================================
def train_horizon_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_val: pd.Series,
    horizon: int,
    logger: logging.Logger,
    seeds: List[int] = [42, 2024, 7, 11, 999]
) -> Tuple[np.ndarray, Dict, List]:
    """Train independent model for a single horizon with multi-seed ensemble.
    
    Returns trained models for reuse in test predictions.
    """
    
    # Horizon-specific hyperparameters (could be tuned independently)
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
    
    val_pred = np.zeros(len(y_val))
    feature_importance = np.zeros(len(X_train.columns))
    trained_models = []
    
    for seed in seeds:
        logger.info(f"  Training seed {seed}...")
        
        mdl = lgb.LGBMRegressor(**lgb_cfg, random_state=seed)
        
        mdl.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        val_pred += mdl.predict(X_val) / len(seeds)
        feature_importance += mdl.feature_importances_ / len(seeds)
        trained_models.append(mdl)
        
        logger.info(f"    Best iteration: {mdl.best_iteration_}")
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return val_pred, {'importance': importance_df, 'config': lgb_cfg}, trained_models


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main pipeline execution with per-horizon IC analysis."""
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("PER-HORIZON IC ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Validation threshold: ts_index > {VAL_THRESHOLD}")
    
    # Compute target encoding stats
    train_stats = compute_train_stats(TRAIN_PATH, VAL_THRESHOLD, logger)
    
    # Storage for results
    cv_cache = {'y': [], 'pred': [], 'wt': []}
    test_outputs = []
    horizon_scores = {}
    horizon_ic_analysis = {}
    feature_importance_all = {}
    
    seeds = [42, 2024, 7, 11, 999]
    logger.info(f"Ensemble seeds: {seeds}")
    
    # Process each horizon independently
    for hz in HORIZONS:
        logger.info("\n" + "=" * 70)
        logger.info(f"HORIZON {hz} - INDEPENDENT MODEL + IC ANALYSIS")
        logger.info("=" * 70)
        
        # Load and filter data
        logger.info(f"Loading horizon {hz} data...")
        tr_df = pd.read_parquet(TRAIN_PATH)
        tr_df = tr_df[tr_df['horizon'] == hz].copy()
        logger.info(f"Train data: {len(tr_df):,} rows")
        
        # Build features
        logger.info("Building context and discovered features...")
        tr_df = build_context_features(tr_df, train_stats, hz, logger)
        
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
        
        # ======== IC ANALYSIS PER FEATURE GROUP ========
        logger.info("\n--- IC/RankIC Analysis per Feature Group ---")
        feature_groups = get_feature_groups(feature_cols)
        
        ic_results = {}
        for group_name, group_cols in feature_groups.items():
            if not group_cols:
                continue
            result = analyze_feature_group_ic(
                tr_df.loc[val_mask],
                group_cols,
                'y_target',
                'weight',
                group_name,
                logger
            )
            ic_results[group_name] = result
            logger.info(
                f"  {group_name:25s}: n={result['n_features']:3d}, "
                f"IC={result['mean_ic']:.4f}, RankIC={result['mean_rank_ic']:.4f}, "
                f"MaxIC={result['max_ic']:.4f} ({result['max_ic_feature']})"
            )
        
        horizon_ic_analysis[hz] = ic_results
        
        # ======== TRAIN HORIZON MODEL ========
        logger.info("\n--- Training Independent Horizon Model ---")
        val_pred, model_info, trained_models = train_horizon_model(
            X_fit, y_fit, w_fit,
            X_hold, y_hold, w_hold,
            hz, logger, seeds
        )
        
        feature_importance_all[hz] = model_info['importance']
        
        # Store CV results
        cv_cache['y'].extend(y_hold.tolist())
        cv_cache['pred'].extend(val_pred.tolist())
        cv_cache['wt'].extend(w_hold.tolist())
        
        # Horizon score
        hz_score = weighted_rmse_score(y_hold, val_pred, w_hold)
        horizon_scores[hz] = hz_score
        logger.info(f"\nHorizon {hz} Score: {hz_score:.5f}")
        
        # ======== TEST PREDICTIONS ========
        if TEST_PATH.exists():
            logger.info("\n--- Generating Test Predictions ---")
            te_df = pd.read_parquet(TEST_PATH)
            te_df = te_df[te_df['horizon'] == hz].copy()
            te_df = build_context_features(te_df, train_stats, hz, logger)
            logger.info(f"Test data: {len(te_df):,} rows")
            
            # Use validation-trained models for test predictions (faster)
            tst_pred = np.zeros(len(te_df))
            for mdl in trained_models:
                tst_pred += mdl.predict(te_df[feature_cols]) / len(trained_models)
            
            test_outputs.append(
                pd.DataFrame({'id': te_df['id'], 'prediction': tst_pred})
            )
            logger.info(f"Test predictions generated for horizon {hz}")
            
            del te_df
        
        # Cleanup
        del tr_df, trained_models
        gc.collect()
    
    # Final combined score
    final_score = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt'])
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"FINAL AGGREGATE SCORE: {final_score:.6f}")
    logger.info("\nPer-Horizon Scores:")
    for hz, score in horizon_scores.items():
        logger.info(f"  Horizon {hz:2d}: {score:.5f}")
    
    # ======== SAVE RESULTS ========
    # Main results file
    results_path = OUTPUTS_DIR / "perhorizon_ic_results.txt"
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PER-HORIZON IC ANALYSIS PIPELINE RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"FINAL AGGREGATE SCORE: {final_score:.6f}\n\n")
        f.write("Per-Horizon Scores:\n")
        for hz, score in horizon_scores.items():
            f.write(f"  Horizon {hz:2d}: {score:.5f}\n")
        
        # IC Analysis per horizon
        for hz in HORIZONS:
            f.write(f"\n{'='*70}\n")
            f.write(f"HORIZON {hz} - IC/RANKIC ANALYSIS BY FEATURE GROUP\n")
            f.write(f"{'='*70}\n\n")
            
            ic_results = horizon_ic_analysis[hz]
            
            # Sort groups by mean IC
            sorted_groups = sorted(
                ic_results.items(),
                key=lambda x: x[1]['mean_ic'],
                reverse=True
            )
            
            f.write(f"{'Group':<30} {'N':>5} {'MeanIC':>10} {'MeanRankIC':>12} {'MaxIC':>10} {'MaxFeature'}\n")
            f.write("-" * 100 + "\n")
            
            for group_name, result in sorted_groups:
                f.write(
                    f"{group_name:<30} {result['n_features']:>5} "
                    f"{result['mean_ic']:>10.4f} {result['mean_rank_ic']:>12.4f} "
                    f"{result['max_ic']:>10.4f} {result['max_ic_feature']}\n"
                )
            
            # Top features by importance
            f.write(f"\n--- Top 15 Features by Importance ---\n")
            imp_df = feature_importance_all[hz]
            for i, row in imp_df.head(15).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.2f}\n")
    
    logger.info(f"Results saved to: {results_path}")
    
    # IC Heatmap data
    heatmap_data = []
    for hz in HORIZONS:
        for group_name, result in horizon_ic_analysis[hz].items():
            heatmap_data.append({
                'horizon': hz,
                'group': group_name,
                'mean_ic': result['mean_ic'],
                'mean_rank_ic': result['mean_rank_ic'],
                'max_ic': result['max_ic'],
                'n_features': result['n_features']
            })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_path = OUTPUTS_DIR / "ic_heatmap_data.csv"
    heatmap_df.to_csv(heatmap_path, index=False)
    logger.info(f"IC heatmap data saved to: {heatmap_path}")
    
    # Feature group ranking
    group_avg = heatmap_df.groupby('group').agg({
        'mean_ic': 'mean',
        'mean_rank_ic': 'mean',
        'n_features': 'mean'
    }).reset_index().sort_values('mean_ic', ascending=False)
    
    ranking_path = OUTPUTS_DIR / "feature_group_ranking.csv"
    group_avg.to_csv(ranking_path, index=False)
    logger.info(f"Feature group ranking saved to: {ranking_path}")
    
    # Save submission CSV
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
        'final_score': final_score,
        'horizon_scores': horizon_scores,
        'ic_analysis': horizon_ic_analysis
    }


if __name__ == "__main__":
    main()
