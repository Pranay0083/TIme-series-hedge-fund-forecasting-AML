import importlib.util
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent
_obj_path = _repo_root / "pipeline" / "deeplearning" / "01_optimization_objective.py"
_spec = importlib.util.spec_from_file_location("optimization_objective", _obj_path)
custom_metric_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(custom_metric_module)

lgbm_weighted_rmse_eval = custom_metric_module.lgbm_weighted_rmse_eval
custom_weighted_rmse_score = custom_metric_module.custom_weighted_rmse_score


class MockDataset:
    def __init__(self, y, w):
        self.y = y
        self.w = w

    def get_label(self):
        return self.y

    def get_weight(self):
        return self.w


y_true = np.random.randn(100)
# Bad predictions
y_pred_bad = np.random.randn(100) * 2
w = np.ones(100)

print(f"Bad Predictions Score: {custom_weighted_rmse_score(y_true, y_pred_bad, w)}")

# Perfect predictions
print(f"Perfect Predictions Score: {custom_weighted_rmse_score(y_true, y_true, w)}")

# Naive baseline (all 0)
y_pred_zero = np.zeros(100)
print(f"Zero Predictions Score: {custom_weighted_rmse_score(y_true, y_pred_zero, w)}")
