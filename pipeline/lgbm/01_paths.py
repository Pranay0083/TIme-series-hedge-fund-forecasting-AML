# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
from pathlib import Path

# pipeline/lgbm -> pipeline -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "combined"
TRAIN_PATH = DATA_RAW_DIR / "train.parquet"
TEST_PATH = DATA_RAW_DIR / "test.parquet"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

HORIZONS = [1, 3, 10, 25]
VAL_THRESHOLD = 3500


def ensure_output_dirs():
    for d in (LOGS_DIR, OUTPUTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_output_dirs()
