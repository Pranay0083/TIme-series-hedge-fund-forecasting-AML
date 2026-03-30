import numpy as np

def feature_neutralization(y_pred, features, proportion=1.0):
    """
    Reduces the linear correlation between the model output and a specific set of features.
    """
    y_pred = y_pred.reshape(-1, 1)
    features = np.nan_to_num(features, nan=0.0)
    
    if len(features) < 2:
        return y_pred.flatten()
        
    scores = np.copy(y_pred)
    exposure = features
    
    try:
        exposure_inv = np.linalg.pinv(exposure)
        beta = np.dot(exposure_inv, scores)
        explained_variance = np.dot(exposure, beta)
        neutralized_scores = scores - proportion * explained_variance
    except np.linalg.LinAlgError:
        neutralized_scores = scores
        
    return neutralized_scores.flatten()

def neutralize_by_cross_section(df, y_pred_col, feature_cols, ts_col='ts_index', proportion=0.5):
    """
    Wrapper method to neutralize predictions exclusively within identical timestep groups.
    """
    df_neutralized = df.copy()
    df_neutralized[f'{y_pred_col}_neutralized'] = df_neutralized[y_pred_col]
    
    for ts, group in df_neutralized.groupby(ts_col):
        preds = group[y_pred_col].values
        feats = group[feature_cols].values
        
        neutral_preds = feature_neutralization(preds, feats, proportion=proportion)
        df_neutralized.loc[group.index, f'{y_pred_col}_neutralized'] = neutral_preds
        
    return df_neutralized
