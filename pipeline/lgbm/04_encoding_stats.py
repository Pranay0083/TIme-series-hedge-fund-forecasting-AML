# ============================================================================
# COMPUTE TARGET ENCODING STATS
# ============================================================================
import gc
from pathlib import Path
from logging import Logger
from typing import Dict

import pandas as pd


def compute_train_stats(train_path: Path, val_threshold: int, logger: Logger) -> Dict:
    """
    Compute target encoding statistics from training data only.

    Uses only data up to val_threshold to avoid leakage.
    """
    logger.info("Computing target encoding statistics...")

    temp = pd.read_parquet(
        train_path,
        columns=["sub_category", "sub_code", "y_target", "ts_index"],
    )

    train_only = temp[temp.ts_index <= val_threshold]

    train_stats = {
        "sub_category": train_only.groupby("sub_category")["y_target"].mean().to_dict(),
        "sub_code": train_only.groupby("sub_code")["y_target"].mean().to_dict(),
        "global_mean": train_only["y_target"].mean(),
    }

    logger.info(f"  sub_category encodings: {len(train_stats['sub_category'])}")
    logger.info(f"  sub_code encodings: {len(train_stats['sub_code'])}")
    logger.info(f"  global_mean: {train_stats['global_mean']:.6f}")

    del temp, train_only
    gc.collect()

    return train_stats
