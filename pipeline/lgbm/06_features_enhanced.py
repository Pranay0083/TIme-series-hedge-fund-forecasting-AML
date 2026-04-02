# ============================================================================
# DISCOVERED + CONTEXT FEATURES (enhanced / IC / sub-category pipelines)
# ============================================================================
from typing import Dict, Optional

import numpy as np
import pandas as pd
from logging import Logger


def build_discovered_features(x: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build interaction features discovered by symbolic regression."""
    eps = 1e-7

    x["gp_l_ca_t"] = x["feature_l"] * x["feature_ca"] * x["feature_t"]
    x["gp_ca_t"] = x["feature_ca"] * x["feature_t"]

    x["psr_bz_t2"] = x["feature_bz"] * (x["feature_t"] ** 2)
    x["psr_bz_div_bp_t2"] = x["feature_bz"] / (x["feature_bp"] / (x["feature_t"] ** 2 + eps) + eps)
    x["psr_vu_abs"] = np.abs(x["feature_v"] * x["feature_u"])
    x["psr_bz_t"] = x["feature_bz"] * x["feature_t"]
    x["psr_neg_s"] = -x["feature_s"]
    x["psr_u_scaled"] = x["feature_u"] * -0.07338854

    x["h1_cd_div_c_q"] = (x["feature_cd"] / (x["feature_c"] + eps)) * x["feature_q"]
    x["h1_aw_inv"] = -7.465855 / (2.7417014 - x["feature_aw"] + eps)
    x["h1_d_s"] = x["feature_d"] * x["feature_s"]
    x["h1_abs_bz"] = np.abs(x["feature_bz"])

    x["h3_bz_triple"] = 3 * x["feature_bz"]
    x["h3_bz_f_ratio"] = x["feature_bz"] + x["feature_bz"] / (0.87934214 - x["feature_f"] + eps)
    x["h3_f_bz"] = (x["feature_f"] - 7.829302) * x["feature_bz"]
    x["h3_bz_f_minus_b_bz"] = (x["feature_bz"] * x["feature_f"]) - (x["feature_b"] * x["feature_bz"])

    x["h10_ah_w_bv"] = x["feature_ah"] + x["feature_w"] * x["feature_bv"]
    x["h10_bs_x_aa"] = (x["feature_bs"] + x["feature_x"]) * x["feature_aa"]
    x["h10_ar_ch"] = x["feature_ar"] - x["feature_ch"]
    x["h10_complex"] = (x["h10_ah_w_bv"] + x["h10_bs_x_aa"] + x["h10_ar_ch"]) * x["feature_bz"]
    x["h10_z_bz2"] = x["feature_z"] * (x["feature_bz"] ** 2)
    x["h10_bz_abs_bz"] = x["feature_bz"] * np.abs(x["feature_bz"])
    x["h10_bz_s"] = x["feature_bz"] * x["feature_s"]
    x["h10_bm_scaled"] = x["feature_bm"] * -0.0021058018

    x["h25_b_bz"] = x["feature_b"] * x["feature_bz"]
    x["h25_bx_bz"] = x["feature_bx"] * x["feature_bz"]
    x["h25_n_bz"] = x["feature_n"] * x["feature_bz"]
    x["h25_o_bz"] = x["feature_o"] + x["feature_bz"]
    x["h25_s_bz_shift"] = x["feature_s"] * (x["feature_bz"] + 1.4804125)
    x["h25_e_bz"] = x["feature_e"] * x["feature_bz"]
    x["h25_cd_abs_bz"] = np.abs(x["feature_cd"]) * x["feature_bz"]

    x["bz_squared"] = x["feature_bz"] ** 2
    x["bz_cubed"] = x["feature_bz"] ** 3
    x["s_squared"] = x["feature_s"] ** 2
    x["t_squared"] = x["feature_t"] ** 2

    x["al_div_am"] = x["feature_al"] / (x["feature_am"] + eps)
    x["bz_div_s"] = x["feature_bz"] / (x["feature_s"] + eps)
    x["cd_div_bz"] = x["feature_cd"] / (x["feature_bz"] + eps)
    x["bz_div_bp"] = x["feature_bz"] / (x["feature_bp"] + eps)

    x["al_bz"] = x["feature_al"] * x["feature_bz"]
    x["al_s"] = x["feature_al"] * x["feature_s"]
    x["v_u"] = x["feature_v"] * x["feature_u"]
    x["cg_by"] = x["feature_cg"] * x["feature_by"]

    return x


def build_context_features(
    data: pd.DataFrame,
    enc_stats: Dict,
    horizon: int,
    logger: Optional[Logger] = None,
) -> pd.DataFrame:
    """Target encoding, lags, rolling, EWM, ranks, and discovered interactions."""
    x = data.copy()
    group_cols = ["code", "sub_code", "sub_category", "horizon"]
    top_features = ["feature_al", "feature_am", "feature_cg", "feature_by", "feature_s", "feature_bz"]

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

    for col in top_features:
        if col not in x.columns:
            continue
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

    for col in ["feature_al", "feature_am", "feature_cg", "feature_by", "d_al_am", "feature_bz"]:
        if col in x.columns:
            x[col + "_rk"] = x.groupby("ts_index")[col].rank(pct=True).astype(np.float32)

    x = build_discovered_features(x, horizon)

    return x
