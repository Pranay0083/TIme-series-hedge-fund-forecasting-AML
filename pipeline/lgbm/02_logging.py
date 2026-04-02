# ============================================================================
# LOGGING SETUP
# ============================================================================
import logging
import sys
from logging import Logger
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def setup_logging(logger_name: str, log_file: str) -> Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(file_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
