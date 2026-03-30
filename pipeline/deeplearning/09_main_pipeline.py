import argparse
import pandas as pd
import numpy as np
import os
import sys
import importlib

# Add current path to import custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def import_local_module(module_name):
    # Dynamic import to handle filenames starting with numbers
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(current_dir, f"{module_name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mod_01 = import_local_module("01_optimization_objective")
mod_03 = import_local_module("03_feature_engineering")
mod_05 = import_local_module("05_estimator_architectures")
mod_07 = import_local_module("07_post_processing_ensembling")
mod_08 = import_local_module("08_pipeline_infrastructure")

custom_weighted_rmse_score = mod_01.custom_weighted_rmse_score
create_spread_ratios = mod_03.create_spread_ratios
create_group_z_scores = mod_03.create_group_z_scores
create_rolling_lag_features = mod_03.create_rolling_lag_features
ExpandingMeanTargetEncoder = mod_03.ExpandingMeanTargetEncoder
HorizonSpecificEstimator = mod_05.HorizonSpecificEstimator
apply_target_clipping = mod_07.apply_target_clipping
aggressive_downcasting = mod_08.aggressive_downcasting
MemoryMonitor = mod_08.MemoryMonitor

def load_data(filepath='train.parquet'):
    """ Loads dataset and aggressively downcasts memory footprint """
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"Failed to load {filepath}. Using a mocked DataFrame for structural testing.")
        np.random.seed(42)
        df = pd.DataFrame({
            'ts_index': np.repeat(np.arange(100), 5), 
            'code': np.tile(np.arange(5), 100), 
            'horizon': np.random.choice([1, 3, 10, 25], 500), 
            'sub_category': np.random.choice([1, 2, 3], 500),
            'feature_al': np.random.randn(500),
            'feature_am': np.random.randn(500),
            'feature_cg': np.random.randn(500),
            'feature_by': np.random.randn(500),
            'y_target': np.random.randn(500)
        })
        df['weight'] = 1.0
        return df

    return aggressive_downcasting(df)

def build_advanced_features(df, target_encoder=None, is_train=True):
    """ Applies theoretical feature engineering functions """
    with MemoryMonitor("Feature Engineering - Spreads & Ratios"):
        df = create_spread_ratios(df)
        
    with MemoryMonitor("Feature Engineering - Group Z-Scores"):
        df = create_group_z_scores(df)
        
    with MemoryMonitor("Feature Engineering - Rolling Lags"):
        df = create_rolling_lag_features(df)
        
    with MemoryMonitor("Feature Engineering - Target Encoding"):
        if 'sub_category' in df.columns:
            if is_train and 'y_target' in df.columns:
                target_encoder = ExpandingMeanTargetEncoder(cat_col='sub_category')
                df = target_encoder.fit_transform_sequential(df, 'y_target')
            elif target_encoder is not None:
                df = target_encoder.transform(df)
            
    return df, target_encoder

def run_evaluation_mode(df):
    """ 
    Trains strictly on 80% chronological data, evaluates on out-of-sample 20%.
    """
    print("--- RUNNING EVALUATION MODE (80/20 SPLIT) ---")
    
    unique_ts = np.sort(df['ts_index'].unique())
    split_idx = int(len(unique_ts) * 0.8)
    train_ts = unique_ts[:split_idx]
    
    # Introduce explicit Gap to prevent lookahead overlap
    gap = 25 
    if split_idx + gap < len(unique_ts):
        eval_ts = unique_ts[split_idx + gap:]
    else:
        eval_ts = unique_ts[split_idx:]
    
    train_df = df[df['ts_index'].isin(train_ts)].copy()
    eval_df = df[df['ts_index'].isin(eval_ts)].copy()
    
    print(f"Train observations: {len(train_df)} | Eval observations: {len(eval_df)}")
    if len(eval_df) == 0:
        print("Not enough timesteps for evaluation. Decrease gap or use larger dataset.")
        return
    
    train_df, target_encoder = build_advanced_features(train_df, is_train=True)
    eval_df, _ = build_advanced_features(eval_df, target_encoder=target_encoder, is_train=False)
    
    # Define features
    exclude_cols = ['y_target', 'weight', 'ts_index', 'horizon', 'code', 'id', 'sub_code', 'sub_category']
    features = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    
    final_preds = np.zeros(len(eval_df))
    eval_df_reset = eval_df.reset_index(drop=True)
    
    for h in [1, 3, 10, 25]:
        print(f"\nEvaluating Horizon: {h}")
        
        trn_h = train_df[train_df['horizon'] == h]
        val_h = eval_df_reset[eval_df_reset['horizon'] == h]
        
        if len(trn_h) == 0 or len(val_h) == 0:
            print(f"Skipping horizon {h} due to missing data.")
            continue
            
        estimator = HorizonSpecificEstimator(horizon=h, model_type='lgb')
        
        X_train, y_train = trn_h[features], trn_h['y_target']
        w_train = trn_h['weight'] if 'weight' in trn_h.columns else np.ones(len(trn_h))
        
        X_val, y_val = val_h[features], val_h['y_target']
        w_val = val_h['weight'] if 'weight' in val_h.columns else np.ones(len(val_h))
        
        estimator.fit(X_train, y_train, w_train, X_val, y_val, w_val)
        
        preds = estimator.predict(X_val)
        
        clipped_preds = apply_target_clipping(preds, y_train.values)
        
        val_indices = val_h.index
        final_preds[val_indices] = clipped_preds
        
    score = custom_weighted_rmse_score(
        y_true=eval_df['y_target'].values,
        y_pred=final_preds,
        sample_weight=eval_df['weight'].values if 'weight' in eval_df.columns else None
    )
    print(f"\nFINAL OOS EVALUATION SKILL SCORE: {score:.5f}")

def run_submission_mode(train_df, test_df=None):
    """
    Trains on 100% of available data. If test_df is provided, outputs submission.csv.
    Regardless, saves trained horizon models to disk.
    """
    print("--- RUNNING SUBMISSION MODE (FULL TRAIN) ---")
    
    train_df, target_encoder = build_advanced_features(train_df, is_train=True)
    exclude_cols = ['y_target', 'weight', 'ts_index', 'horizon', 'code', 'id', 'sub_code', 'sub_category']
    features = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    
    # Check if test_df contains the exact random mock string to skip
    has_real_test = False
    if test_df is not None and not (len(test_df) == 500 and 'code' in test_df and test_df['code'].max() == 4):
        has_real_test = True
        test_df, _ = build_advanced_features(test_df, target_encoder=target_encoder, is_train=False)
        submission_preds = np.zeros(len(test_df))
        test_df_reset = test_df.reset_index(drop=True)
    
    os.makedirs(os.path.join(current_dir, 'models'), exist_ok=True)
    print(f"Features mapped ({len(features)} dims). Commencing full training...")

    for h in [1, 3, 10, 25]:
        print(f"\nTraining Full Horizon: {h}")
        trn_h = train_df[train_df['horizon'] == h]
        
        if len(trn_h) == 0:
            print(f"Skipping horizon {h} due to missing training data.")
            continue
            
        estimator = HorizonSpecificEstimator(horizon=h, model_type='lgb')
        X_train, y_train = trn_h[features], trn_h['y_target']
        w_train = trn_h['weight'] if 'weight' in trn_h.columns else np.ones(len(trn_h))
        
        estimator.fit(X_train, y_train, w_train)
        
        # Save Model Weights
        model_path = os.path.join(current_dir, 'models', f'lgb_horizon_{h}.txt')
        estimator.save_model(model_path)
        print(f"Saved model to: {model_path}")
        
        # Process Test Inferences if real test_data exists
        if has_real_test:
            tst_h = test_df_reset[test_df_reset['horizon'] == h]
            if len(tst_h) > 0:
                X_test = tst_h[features]
                preds = estimator.predict(X_test)
                clipped_preds = apply_target_clipping(preds, y_train.values)
                test_indices = tst_h.index
                submission_preds[test_indices] = clipped_preds
        
    if has_real_test:
        submission = pd.DataFrame({
            'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
            'prediction': submission_preds
        })
        submission.to_csv('submission.csv', index=False)
        print("\nSuccessfully generated submission.csv")
    else:
        print("\nNo explicit test data given. Models trained and saved exclusively.")

def run_inference_mode(train_df, test_df):
    """ Loads saved models from disk and infers over the test_df without retraining. """
    print("--- RUNNING INFERENCE MODE (USING SAVED MODELS) ---")
    
    train_df, target_encoder = build_advanced_features(train_df, is_train=True)
    test_df, _ = build_advanced_features(test_df, target_encoder=target_encoder, is_train=False)
    
    exclude_cols = ['y_target', 'weight', 'ts_index', 'horizon', 'code', 'id', 'sub_code', 'sub_category']
    features = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    
    submission_preds = np.zeros(len(test_df))
    test_df_reset = test_df.reset_index(drop=True)
    
    print(f"Features mapped ({len(features)} dims). Commencing inference...")
    
    for h in [1, 3, 10, 25]:
        print(f"\nInferencing Horizon: {h}")
        tst_h = test_df_reset[test_df_reset['horizon'] == h]
        
        if len(tst_h) == 0:
            print(f"Skipping horizon {h} due to missing test data.")
            continue
            
        estimator = HorizonSpecificEstimator(horizon=h, model_type='lgb')
        model_path = os.path.join(current_dir, 'models', f'lgb_horizon_{h}.txt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved model missing: {model_path}. You must run --mode submit first.")
            
        estimator.load_model(model_path)
        
        X_test = tst_h[features]
        preds = estimator.predict(X_test)
        
        # Pull original y sequence stats to calculate clip constraints
        trn_h = train_df[train_df['horizon'] == h]
        y_train_vals = trn_h['y_target'].values if 'y_target' in trn_h.columns else np.array([])
        clipped_preds = apply_target_clipping(preds, y_train_vals)
        
        test_indices = tst_h.index
        submission_preds[test_indices] = clipped_preds
        
    submission = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'prediction': submission_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\nSuccessfully generated submission.csv via fast inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Forecasting Gemini Pipeline")
    parser.add_argument('--mode', type=str, choices=['eval', 'submit', 'infer'], default='eval', help='Mode to run: eval, submit, or infer')
    parser.add_argument('--train_data', type=str, default='train.parquet', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='test.parquet', help='Path to test data for submission')
    
    args = parser.parse_args()
    
    df_train = load_data(args.train_data)
    
    if args.mode == 'eval':
        run_evaluation_mode(df_train)
    elif args.mode == 'submit':
        df_test = load_data(args.test_data)
        run_submission_mode(df_train, df_test)
    elif args.mode == 'infer':
        df_test = load_data(args.test_data)
        run_inference_mode(df_train, df_test)
