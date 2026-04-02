# ============================================================================
# WEIGHTED RMSE SCORE (Competition Metric)
# ============================================================================
import numpy as np


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
