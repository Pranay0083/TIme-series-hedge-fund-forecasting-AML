# ============================================================================
# TEMPORAL DECAY WEIGHTS + DECAY-AWARE HORIZON TRAINING
# ============================================================================
from logging import Logger
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

DECAY_HALF_LIVES = {
    1: 500,
    3: 700,
    10: 1000,
    25: 1500,
}


def compute_decay_weights(ts_index: pd.Series, half_life: int, logger: Logger) -> np.ndarray:
    """Exponential temporal decay weights in [0, 1]."""
    ts = ts_index.values.astype(np.float64)
    max_ts = ts.max()
    age = max_ts - ts

    lam = np.log(2) / half_life
    decay = np.exp(-lam * age)

    logger.info(f"  Decay half-life: {half_life}")
    logger.info(f"  ts_index range: [{ts.min():.0f}, {max_ts:.0f}]")
    logger.info(f"  Age range: [{age.min():.0f}, {age.max():.0f}]")
    logger.info(f"  Decay factor range: [{decay.min():.6f}, {decay.max():.6f}]")
    logger.info(f"  Decay factor mean: {decay.mean():.6f}")
    logger.info(f"  Effective sample count (sum of weights): {decay.sum():.1f} / {len(decay)}")

    return decay


def train_horizon_model_decay(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    decay_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_val: pd.Series,
    decay_val: np.ndarray,
    horizon: int,
    logger: Logger,
    seeds: List[int] = None,
) -> Tuple[np.ndarray, Dict, List]:
    """Multi-seed ensemble; sample_weight = competition_weight × decay_factor."""
    if seeds is None:
        seeds = [42, 2024, 7, 11, 999]

    eff_train = w_train.values * decay_train
    eff_val = w_val.values * decay_val

    logger.info(
        f"  Effective train weights: min={eff_train.min():.4f}, "
        f"max={eff_train.max():.4f}, mean={eff_train.mean():.4f}"
    )
    logger.info(
        f"  Effective val weights:   min={eff_val.min():.4f}, "
        f"max={eff_val.max():.4f}, mean={eff_val.mean():.4f}"
    )

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
            sample_weight=eff_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[eff_val],
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
