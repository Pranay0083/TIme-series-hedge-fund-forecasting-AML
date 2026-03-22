"""Time Series Cross-Validation pipeline for Hedge Fund Forecasting AML.

Usage:
    from pipeline.timeseries_split import TimeSeriesCVPipeline
    cv = TimeSeriesCVPipeline(test_size=180, n_splits=5, gap_size=5)
    results = cv.run_cv(df, model_fn=my_model_fn)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit


class TimeSeriesCVPipeline:

    def __init__(self, test_size=180, n_splits=5, gap_size=5,
                 window_type="expanding", time_col="ts_index",
                 target_col="y_target_hnorm", weight_col="weight"):
        self.test_size = test_size
        self.n_splits = n_splits
        self.gap_size = gap_size
        self.window_type = window_type
        self.time_col = time_col
        self.target_col = target_col
        self.weight_col = weight_col

    def get_cv_args(self):
        return {
            "test_size": self.test_size,
            "n_splits": self.n_splits,
            "gap_size": self.gap_size,
            "window_type": self.window_type,
        }

    @staticmethod
    def get_feature_cols(df):
        exclude = {"id", "code", "sub_code", "sub_category", "horizon",
                   "ts_index", "y_target", "y_target_clipped",
                   "y_target_hnorm", "weight"}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    @staticmethod
    def weighted_rmse(y_true, y_pred, weights):
        w = weights / weights.sum()
        return np.sqrt(np.sum(w * (y_true - y_pred) ** 2))

    @staticmethod
    def spearman_per_date(df, pred_col="prediction",
                          target_col="y_target_hnorm", time_col="ts_index"):
        correlations = []
        for ts in df[time_col].unique():
            subset = df[df[time_col] == ts]
            if len(subset) < 3:
                continue
            corr, _ = spearmanr(subset[pred_col].values, subset[target_col].values)
            if not np.isnan(corr):
                correlations.append(corr)
        return np.mean(correlations) * 100 if correlations else 0.0

    def run_cv(self, df, model_fn=None, verbose=True):
        """Run time-series cross-validation.

        Args:
            df: preprocessed DataFrame
            model_fn: callable(X_train, y_train) -> fitted model with .predict()
                      Defaults to LinearRegression.
            verbose: print fold-level results

        Returns:
            list of dicts with fold-level metrics
        """
        if model_fn is None:
            def model_fn(X_train, y_train):
                m = LinearRegression()
                m.fit(X_train, y_train)
                return m

        feature_cols = self.get_feature_cols(df)
        groups = df[self.time_col].values
        CV = GroupTimeSeriesSplit(**self.get_cv_args())
        results = []

        for fold_i, (train_idx, val_idx) in enumerate(
            CV.split(df, groups=groups)
        ):
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()

            X_train = train_fold[feature_cols].fillna(0).values
            y_train = train_fold[self.target_col].fillna(0).values
            X_val = val_fold[feature_cols].fillna(0).values
            y_val = val_fold[self.target_col].values

            model = model_fn(X_train, y_train)
            y_pred = model.predict(X_val)

            val_fold = val_fold.copy()
            val_fold["prediction"] = y_pred

            w = val_fold[self.weight_col].values if self.weight_col in val_fold.columns else np.ones(len(y_val))
            w_rmse = self.weighted_rmse(y_val, y_pred, w)
            spearman = self.spearman_per_date(
                val_fold, target_col=self.target_col, time_col=self.time_col
            )

            results.append({
                "fold": fold_i + 1,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "weighted_rmse": w_rmse,
                "spearman_pct": spearman,
            })

            if verbose:
                print(f"  Fold {fold_i+1}: wRMSE={w_rmse:.6f}, Spearman={spearman:.2f}%")

        return results
