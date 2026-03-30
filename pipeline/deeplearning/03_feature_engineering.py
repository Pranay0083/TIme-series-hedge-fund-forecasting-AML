import pandas as pd
import numpy as np

def create_spread_ratios(df):
    """
    Computes difference spreads and imbalance ratios for highly predictive anonymized features.
    """
    df = df.copy()
    epsilon = 1e-8
    
    if all(col in df.columns for col in ['feature_al', 'feature_am', 'feature_cg', 'feature_by']):
        df['spread_al_am'] = df['feature_al'] - df['feature_am']
        df['spread_cg_by'] = df['feature_cg'] - df['feature_by']
        
        df['ratio_al_am'] = df['feature_al'] / (df['feature_am'] + epsilon)
        df['imbalance_cg_by'] = (df['feature_cg'] - df['feature_by']) / (df['feature_cg'] + df['feature_by'] + epsilon)
        
    return df

def create_group_z_scores(df, groupby_col='sub_category', target_cols=['feature_al', 'feature_am']):
    """
    Computes group-based relative Z-scores to contextualize isolated feature vectors.
    """
    df = df.copy()
    for col in target_cols:
        if col in df.columns:
            grouped = df.groupby(['ts_index', groupby_col])[col]
            cross_mean = grouped.transform('mean')
            cross_std = grouped.transform('std')
            df[f'z_score_{col}_{groupby_col}'] = (df[col] - cross_mean) / (cross_std + 1e-8)
            
    return df

def create_rolling_lag_features(df, target_cols=['feature_al', 'spread_al_am'], lags=[1, 3, 5]):
    """
    Computes strict rolling lag features grouped by individual unique entity to prevent cross-sectional leakage.
    Ensures that original DataFrame row order is preserved.
    """
    original_idx = df.index
    df_sorted = df.sort_values(by=['code', 'ts_index'])
    
    for col in target_cols:
        if col in df_sorted.columns:
            for lag in lags:
                df_sorted[f'lag_{lag}_{col}'] = df_sorted.groupby('code')[col].shift(lag)
                
    return df_sorted.loc[original_idx]

class ExpandingMeanTargetEncoder:
    """
    Implements a strict sequential expanding mean target encoder.
    Optimized to be fully vectorized across ts_index.
    """
    def __init__(self, cat_col, lambda_smoothing=50):
        self.cat_col = cat_col
        self.lambda_smoothing = lambda_smoothing
        self.global_mean = 0.0
        self.terminal_sums = {}
        self.terminal_counts = {}
        
    def fit_transform_sequential(self, df, target_col):
        """ Processes sequential dataframe, mapping categorical value at time t strictly on time < t. """
        df = df.copy()
        original_idx = df.index
        df_sorted = df.sort_values('ts_index')
        
        # Calculate shifting global mean
        expanding_global_mean = df_sorted[target_col].shift(1).expanding().mean()
        df_sorted['__global_mean'] = expanding_global_mean.fillna(0)
        
        # Calculate shifting category mean
        df_sorted['__shifted_target'] = df_sorted.groupby(self.cat_col)[target_col].shift(1)
        cat_cumsum = df_sorted.groupby(self.cat_col)['__shifted_target'].cumsum().fillna(0)
        cat_cumcount = df_sorted.groupby(self.cat_col).cumcount() # 0 for first, 1 for second... matches perfectly!
        
        # Ensure division by zero is handled
        cat_cumcount_safe = np.maximum(cat_cumcount, 1)
        cat_mean = cat_cumsum / cat_cumcount_safe
        
        # Bayesian Smoothing
        weight = 1.0 / (1.0 + np.exp(-(cat_cumcount - self.lambda_smoothing) / 10.0)) 
        
        encoded_vals = cat_mean * weight + df_sorted['__global_mean'] * (1 - weight)
        
        # Set 0 variance defaults for the first ever observation (count=0)
        encoded_vals = encoded_vals.where(cat_cumcount > 0, df_sorted['__global_mean'])
        
        col_name = f'{self.cat_col}_target_encoded'
        df_sorted[col_name] = encoded_vals
        
        # Save exact terminal states for test transform
        self.global_mean = df_sorted[target_col].mean()
        self.terminal_sums = df_sorted.groupby(self.cat_col)[target_col].sum().to_dict()
        self.terminal_counts = df_sorted.groupby(self.cat_col).size().to_dict()
        
        # Restore index order
        df_sorted = df_sorted.loc[original_idx]
        df[col_name] = df_sorted[col_name]
        df = df.drop(columns=['__shifted_target'], errors='ignore')
        return df

    def transform(self, df):
        """ Transform a test dataframe relying exclusively on the frozen terminal state. """
        df = df.copy()
        encoded_vals = []
        for cat in df[self.cat_col]:
            cat_sum = self.terminal_sums.get(cat, 0.0)
            cat_count = self.terminal_counts.get(cat, 0)
            
            if cat_count == 0:
                encoded_vals.append(self.global_mean)
                continue
                
            cat_mean = cat_sum / cat_count
            weight = 1.0 / (1.0 + np.exp(-(cat_count - self.lambda_smoothing) / 10.0)) 
            val = cat_mean * weight + self.global_mean * (1 - weight)
            encoded_vals.append(val)
            
        df[f'{self.cat_col}_target_encoded'] = encoded_vals
        return df
