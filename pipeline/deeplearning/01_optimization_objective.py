import numpy as np

def custom_weighted_rmse_score(y_true, y_pred, sample_weight):
    """
    Calculates the exact leaderboard Custom Weighted RMSE Skill Score.
    """
    if len(y_true) == 0:
        return 0.0
        
    if sample_weight is None or len(sample_weight) != len(y_true):
        sample_weight = np.ones_like(y_true)
        
    numerator = np.sum(sample_weight * (y_true - y_pred)**2)
    denominator = np.sum(sample_weight * (y_true**2))
    
    if denominator == 0:
        ratio = 0.0
    else:
        ratio = numerator / denominator
        
    ratio = np.clip(ratio, 0.0, 1.0)
    score = np.sqrt(1.0 - ratio)
    return score

def lgbm_weighted_rmse_eval(y_pred, dataset):
    """ Custom evaluation metric for LightGBM. """
    y_true = dataset.get_label()
    weight = dataset.get_weight()
    
    if weight is None or len(weight) != len(y_true):
        weight = np.ones_like(y_true)
        
    epsilon = 1e-8
    numerator = np.sum(weight * (y_true - y_pred)**2)
    denominator = np.sum(weight * (y_true**2)) + epsilon
    
    ratio = np.clip(numerator / denominator, 0.0, 1.0)
    score = np.sqrt(1.0 - ratio)
    
    # We want to maximize the skill score
    return 'weighted_rmse', score, True

def xgb_weighted_rmse_eval(y_pred, dtrain):
    """ Custom evaluation metric for XGBoost. """
    y_true = dtrain.get_label()
    weight = dtrain.get_weight()
    
    if weight is None or len(weight) != len(y_true):
        weight = np.ones_like(y_true)
        
    epsilon = 1e-8
    numerator = np.sum(weight * (y_true - y_pred)**2)
    denominator = np.sum(weight * (y_true**2)) + epsilon
    
    ratio = np.clip(numerator / denominator, 0.0, 1.0)
    score = np.sqrt(1.0 - ratio)
    
    return 'weighted_rmse', score
