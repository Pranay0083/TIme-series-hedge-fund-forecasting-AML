# ============================================================================
# FEATURE GROUP IC ANALYSIS + STANDARD HORIZON TRAINING
# ============================================================================
from logging import Logger
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats


def get_feature_groups(all_columns: List[str]) -> Dict[str, List[str]]:
    """Define feature groups for IC analysis."""
    groups = {
        "category_encodings": [c for c in all_columns if c.endswith("_enc")],
        "volatility_stats": [c for c in all_columns if "rollstd" in c or "diff1" in c],
        "lag_features": [c for c in all_columns if "_lag" in c],
        "rolling_features": [c for c in all_columns if "_roll" in c and "std" not in c],
        "ewm_features": [c for c in all_columns if "_ewm" in c],
        "rank_features": [c for c in all_columns if "_rk" in c],
        "discovered_gp_psr": [c for c in all_columns if c.startswith("gp_") or c.startswith("psr_")],
        "discovered_horizon": [c for c in all_columns if c.startswith(("h1_", "h3_", "h10_", "h25_"))],
        "polynomial": [c for c in all_columns if c in ["bz_squared", "bz_cubed", "s_squared", "t_squared"]],
        "ratios": [c for c in all_columns if c in ["al_div_am", "bz_div_s", "cd_div_bz", "bz_div_bp"]],
        "raw_signals": [c for c in all_columns if c.startswith("feature_")],
    }
    return {k: v for k, v in groups.items() if len(v) > 0}


def compute_ic(feature: np.ndarray, target: np.ndarray) -> float:
    mask = ~(np.isnan(feature) | np.isnan(target))
    if mask.sum() < 10:
        return np.nan
    f, t = feature[mask], target[mask]
    if np.std(f) < 1e-10 or np.std(t) < 1e-10:
        return 0.0
    return np.corrcoef(f, t)[0, 1]


def compute_rank_ic(feature: np.ndarray, target: np.ndarray) -> float:
    mask = ~(np.isnan(feature) | np.isnan(target))
    if mask.sum() < 10:
        return np.nan
    f, t = feature[mask], target[mask]
    corr, _ = stats.spearmanr(f, t)
    return corr


def compute_weighted_ic(feature: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    mask = ~(np.isnan(feature) | np.isnan(target) | np.isnan(weights))
    if mask.sum() < 10:
        return np.nan
    f, t, w = feature[mask], target[mask], weights[mask]

    w_sum = w.sum()
    f_mean = (w * f).sum() / w_sum
    t_mean = (w * t).sum() / w_sum

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
    group_name: str,
    logger: Logger,
    *,
    weight_col: Optional[str] = None,
    ic_sample_weight: Optional[np.ndarray] = None,
) -> Dict:
    """
    IC metrics for a feature group.

    Either weight_col (column name for competition weights) or ic_sample_weight
    (aligned array for decay-weighted IC) must be provided.
    """
    if weight_col is not None and ic_sample_weight is not None:
        raise ValueError("Provide only one of weight_col or ic_sample_weight")
    if weight_col is None and ic_sample_weight is None:
        raise ValueError("Provide weight_col or ic_sample_weight")

    target = df[target_col].values
    if weight_col is not None:
        weights = df[weight_col].values
    else:
        weights = ic_sample_weight

    results = {
        "group": group_name,
        "n_features": len(feature_cols),
        "features": {},
        "mean_ic": 0.0,
        "mean_rank_ic": 0.0,
        "mean_weighted_ic": 0.0,
        "max_ic": 0.0,
        "max_ic_feature": None,
    }

    ics, rank_ics, weighted_ics = [], [], []

    for col in feature_cols:
        if col not in df.columns:
            continue
        feature = df[col].values

        ic = compute_ic(feature, target)
        rank_ic = compute_rank_ic(feature, target)
        w_ic = compute_weighted_ic(feature, target, weights)

        results["features"][col] = {
            "ic": ic,
            "rank_ic": rank_ic,
            "weighted_ic": w_ic,
        }

        if not np.isnan(ic):
            ics.append(abs(ic))
        if not np.isnan(rank_ic):
            rank_ics.append(abs(rank_ic))
        if not np.isnan(w_ic):
            weighted_ics.append(abs(w_ic))

    if ics:
        results["mean_ic"] = np.mean(ics)
        results["max_ic"] = max(ics)
        max_idx = ics.index(max(ics))
        results["max_ic_feature"] = list(results["features"].keys())[max_idx]
    if rank_ics:
        results["mean_rank_ic"] = np.mean(rank_ics)
    if weighted_ics:
        results["mean_weighted_ic"] = np.mean(weighted_ics)

    return results


def train_horizon_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_val: pd.Series,
    horizon: int,
    logger: Logger,
    seeds: List[int] = None,
) -> Tuple[np.ndarray, Dict, List]:
    """Train independent model for a single horizon with multi-seed ensemble."""
    if seeds is None:
        seeds = [42, 2024, 7, 11, 999]

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
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        val_pred += mdl.predict(X_val) / len(seeds)
        feature_importance += mdl.feature_importances_ / len(seeds)
        trained_models.append(mdl)

        logger.info(f"    Best iteration: {mdl.best_iteration_}")

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False)

    return val_pred, {"importance": importance_df, "config": lgb_cfg}, trained_models
