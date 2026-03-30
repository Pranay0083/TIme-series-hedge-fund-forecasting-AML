import numpy as np

def seed_average_predictions(predictions_list):
    """
    Linearly averages raw predictions from N structurally identical models 
    trained under distinct initialization seeds.
    """
    if not predictions_list:
        return np.array([])
        
    stacked_preds = np.stack(predictions_list, axis=0) # Shape: (Num_Seeds, Num_Samples)
    return np.mean(stacked_preds, axis=0) # Shape: (Num_Samples)

def linear_weighted_blend(pred_dict, weights_dict):
    """
    Blends independent out-of-fold/test prediction arrays via learned fractional weights.
    
    pred_dict: {'lgbm': array([1,2,3]), 'xgb': array([1.1, 1.9, 3.1]) }
    weights_dict: {'lgbm': 0.55, 'xgb': 0.45}
    """
    assert set(pred_dict.keys()) == set(weights_dict.keys()), "Mismatched models in blend."
    
    final_blend = np.zeros_like(list(pred_dict.values())[0])
    
    for model_name, preds in pred_dict.items():
        final_blend += preds * weights_dict[model_name]
        
    return final_blend

def apply_target_clipping(y_pred, y_target_train, lower_percentile=1.0, upper_percentile=99.0):
    """
    Rigidly bounds output predictions based strictly on training set percentiles
    to negate extreme asymmetric penalty risks on unpredictable outliers.
    """
    if len(y_target_train) == 0:
        return y_pred
        
    p_lower = np.percentile(y_target_train, lower_percentile)
    p_upper = np.percentile(y_target_train, upper_percentile)
    
    return np.clip(y_pred, p_lower, p_upper)
