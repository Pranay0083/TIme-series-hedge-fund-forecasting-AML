# ============================================================================
# PATHS CONFIGURATION (same layout as pipeline/lgbm/01_paths)
# ============================================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "combined"
TRAIN_PATH = DATA_RAW_DIR / "train.parquet"
TEST_PATH = DATA_RAW_DIR / "test.parquet"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DL_MODELS_DIR = Path(__file__).resolve().parent / "models"

HORIZONS = [1, 3, 10, 25]
VAL_THRESHOLD = 3500


def ensure_output_dirs():
    for d in (LOGS_DIR, OUTPUTS_DIR, MODELS_DIR, DL_MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_output_dirs()
