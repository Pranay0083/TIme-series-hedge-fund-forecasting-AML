"""Preprocessing pipeline for Hedge Fund Forecasting AML.

Usage:
    from pipeline.preprocess import PreprocessPipeline
    pp = PreprocessPipeline.from_config('pipeline/preprocess_config.json',
                                        'pipeline/group_medians.parquet')
    train_clean = pp.transform_train(train_raw)
    test_clean  = pp.transform_test(test_raw)
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PreprocessPipeline:

    def __init__(self, config: dict, group_medians: pd.DataFrame):
        self.config = config
        self.group_medians = group_medians
        self.global_medians = pd.Series(config['global_medians'])
        self.label_encoders = {}

    @classmethod
    def from_config(cls, config_path: str, group_medians_path: str):
        with open(config_path) as f:
            config = json.load(f)
        group_medians = pd.read_parquet(group_medians_path)
        return cls(config, group_medians)

    def _drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.config['features_to_drop'], errors='ignore')

    def _add_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        for f in self.config['high_missing_features']:
            if f in df.columns:
                df[f'{f}_is_missing'] = df[f].isnull().astype(np.int8)
        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = ['code', 'sub_category']
        cols = [c for c in self.group_medians.columns if c in df.columns]
        for col in cols:
            mask = df[col].isnull()
            if mask.sum() == 0:
                continue
            fill = df.loc[mask, group_cols].merge(
                self.group_medians[[col]].reset_index(), on=group_cols, how='left'
            )[col]
            fill.index = df.loc[mask].index
            df.loc[mask, col] = fill.fillna(
                self.global_medians[col] if col in self.global_medians.index else 0
            )
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        for col in self.config['cat_cols']:
            if fit or col not in self.label_encoders:
                le = LabelEncoder()
                le.fit(df[col].unique())
                self.label_encoders[col] = le
            le = self.label_encoders[col]
            known = set(le.classes_)
            df[f'{col}_encoded'] = df[col].apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )
        return df

    def _process_target(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df['y_target_clipped'] = df['y_target'].clip(
            cfg['target_clip_lower'], cfg['target_clip_upper']
        )
        h_stats = pd.DataFrame(cfg['horizon_stats'])
        h_stats.index = h_stats.index.astype(df['horizon'].dtype)
        df = df.merge(h_stats, left_on='horizon', right_index=True, how='left')
        df['y_target_hnorm'] = (df['y_target_clipped'] - df['h_mean']) / df['h_std']
        df.drop(columns=['h_mean', 'h_std'], inplace=True, errors='ignore')
        return df

    def _drop_zero_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['weight'] > 0].reset_index(drop=True)
        return df

    def transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._drop_features(df)
        df = self._add_missing_flags(df)
        df = self._impute(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._process_target(df)
        df = self._drop_zero_weight(df)
        return df

    def transform_test(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._drop_features(df)
        df = self._add_missing_flags(df)
        df = self._impute(df)
        df = self._encode_categoricals(df, fit=False)
        return df