# ============================================================================
# CONTEXT FEATURE ENGINEERING (advanced pipeline)
# ============================================================================
from typing import Dict, Optional

import numpy as np
import pandas as pd
from logging import Logger


def build_context_features(
    data: pd.DataFrame,
    enc_stats: Dict = None,
    logger: Optional[Logger] = None,
) -> pd.DataFrame:
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
    group_cols = ["code", "sub_code", "sub_category", "horizon"]
    top_features = ["feature_al", "feature_am", "feature_cg", "feature_by", "feature_s"]

    if enc_stats is not None:
        for c in ["sub_category", "sub_code"]:
            x[c + "_enc"] = x[c].map(enc_stats[c]).fillna(enc_stats["global_mean"])

    x["d_al_am"] = x["feature_al"] - x["feature_am"]
    x["r_al_am"] = x["feature_al"] / (x["feature_am"] + 1e-7)
    x["d_cg_by"] = x["feature_cg"] - x["feature_by"]

    for col in top_features:
        if col not in x.columns:
            continue

        for lag in [1, 3, 10]:
            x[f"{col}_lag{lag}"] = x.groupby(group_cols)[col].shift(lag).astype(np.float32)

        x[f"{col}_diff1"] = x.groupby(group_cols)[col].diff(1).astype(np.float32)

        for window in [5, 10]:
            x[f"{col}_roll{window}"] = x.groupby(group_cols)[col].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            ).astype(np.float32)

            x[f"{col}_rollstd{window}"] = x.groupby(group_cols)[col].transform(
                lambda s: s.rolling(window, min_periods=1).std()
            ).astype(np.float32)

        x[f"{col}_ewm5"] = x.groupby(group_cols)[col].transform(
            lambda s: s.ewm(span=5, adjust=False).mean()
        ).astype(np.float32)

    x["t_cycle"] = np.sin(2 * np.pi * x["ts_index"] / 100)

    for col in ["feature_al", "feature_am", "feature_cg", "feature_by", "d_al_am"]:
        if col in x.columns:
            x[col + "_rk"] = x.groupby("ts_index")[col].rank(pct=True).astype(np.float32)

    return x
