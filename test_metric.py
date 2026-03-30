import numpy as np
import importlib

custom_metric_module = importlib.import_module("pipeline.gemini_research.01_optimization_objective")
lgbm_weighted_rmse_eval = custom_metric_module.lgbm_weighted_rmse_eval
custom_weighted_rmse_score = custom_metric_module.custom_weighted_rmse_score

class MockDataset:
    def __init__(self, y, w):
        self.y = y
        self.w = w
    def get_label(self): return self.y
    def get_weight(self): return self.w

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
