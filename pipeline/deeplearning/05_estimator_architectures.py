import importlib.util
import os

import lightgbm as lgb
import numpy as np
import xgboost as xgb

spec = importlib.util.spec_from_file_location(
    "01_optimization_objective",
    os.path.join(os.path.dirname(__file__), "01_optimization_objective.py"),
)
custom_metric_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_metric_module)
lgbm_weighted_rmse_eval = custom_metric_module.lgbm_weighted_rmse_eval
xgb_weighted_rmse_eval = custom_metric_module.xgb_weighted_rmse_eval

class HorizonSpecificEstimator:
    """ Wrapper class to train and infer independent models partitioned strictly by horizon. """
    def __init__(self, horizon, model_type='lgb', random_seed=42):
        self.horizon = horizon
        self.model_type = model_type
        self.random_seed = random_seed
        self.model = None
        
        if self.model_type == 'lgb':
            self.params = {
                'objective': 'regression', 
                'metric': 'None',
                'learning_rate': 0.02,
                'num_leaves': 31 if horizon < 10 else 64,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'extra_trees': True,
                'min_data_in_leaf': 100,
                'seed': self.random_seed,
                'verbosity': -1,
                'lambda_l2': 1.0 if horizon < 10 else 5.0
            }
        elif self.model_type == 'xgb':
            self.params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.02,
                'max_depth': 5 if horizon < 10 else 6,
                'subsample': 0.8,
                'colsample_bytree': 0.6,
                'reg_lambda': 1.0 if horizon < 10 else 5.0,
                'seed': self.random_seed,
                'tree_method': 'hist'
            }
            
    def fit(self, X_train, y_train, w_train=None, X_val=None, y_val=None, w_val=None):
        if self.model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
            valid_data = None
            if X_val is not None:
                valid_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)
                
            callbacks = [lgb.log_evaluation(period=100)]
            if valid_data:
                callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))
                
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=3000,
                valid_sets=[train_data, valid_data] if valid_data else [train_data],
                feval=lgbm_weighted_rmse_eval,
                callbacks=callbacks
            )
            
        elif self.model_type == 'xgb':
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
            
            dval = None
            evals = [(dtrain, 'train')]
            if X_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
                evals.append((dval, 'valid'))
                
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=3000,
                evals=evals,
                early_stopping_rounds=100 if dval else None,
                verbose_eval=100,
                feval=xgb_weighted_rmse_eval,
                custom_metric=xgb_weighted_rmse_eval
            )
            
    def predict(self, X):
        if self.model is None:
            raise ValueError("Estimator is not fitted yet.")
            
        if self.model_type == 'lgb':
            return self.model.predict(X, num_iteration=self.model.best_iteration)
        elif self.model_type == 'xgb':
            dtest = xgb.DMatrix(X)
            return self.model.predict(dtest, iteration_range=(0, self.model.best_iteration))

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Estimator is not fitted yet.")
        self.model.save_model(filepath)
        
    def load_model(self, filepath):
        if self.model_type == 'lgb':
            self.model = lgb.Booster(model_file=filepath)
        elif self.model_type == 'xgb':
            self.model = xgb.Booster(model_file=filepath)
