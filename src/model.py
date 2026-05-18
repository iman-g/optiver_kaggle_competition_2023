# src/model.py
import numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

from src.config import CONFIG, XGB_PARAMS, LGB_PARAMS


class PurgedGroupTimeSeriesSplit:
    """
    Time-series cross-validator that prevents data leakage.

    Keeps chronological order (train always before validation) and removes
    a purge gap between train and validation to prevent lag features from
    leaking future information.
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, df, groups):
        unique_groups = np.sort(groups.unique())
        n_groups = len(unique_groups)
        fold_size = n_groups // (self.n_splits + 1)

        for i in range(self.n_splits):
            val_start = (i + 1) * fold_size
            val_end = val_start + fold_size
            val_groups = unique_groups[val_start:val_end]

            train_end = val_start - self.purge_gap
            if train_end <= 0:
                continue
            train_groups = unique_groups[:train_end]

            train_idx = groups.isin(train_groups)
            val_idx = groups.isin(val_groups)

            yield train_idx, val_idx


def prepare_train_val(df, train_idx, val_idx, stock_profiles):
    """Split data into train/val arrays and return feature column names."""
    train_df = df[train_idx].copy()
    val_df = df[val_idx].copy()

    # Clip extreme targets to reduce noise from outliers
    q_low = train_df['target'].quantile(0.001)
    q_high = train_df['target'].quantile(0.999)
    train_df['target'] = train_df['target'].clip(q_low, q_high)

    drop_cols = ['row_id', 'time_id', 'date_id', 'target', 'stock_id', 'currently_scored']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']

    return X_train, y_train, X_val, y_val, val_df, feature_cols


def weighted_zero_sum_adjustment(predictions: np.ndarray, val_df, stock_weights: dict) -> np.ndarray:
    """
    Enforce the zero-sum constraint implied by the target definition.

    The target measures each stock's movement relative to a weighted index.
    The weighted sum of all targets per timestamp must therefore equal zero.
    Raw model predictions violate this; subtracting the weighted mean per
    timestamp restores the constraint at no cost to model training.
    """
    import pandas as pd

    temp_df = val_df[['date_id', 'seconds_in_bucket', 'stock_id']].copy().reset_index(drop=True)
    temp_df['pred'] = predictions.copy()
    temp_df['weight'] = temp_df['stock_id'].map(stock_weights).fillna(0)
    temp_df['weighted_pred'] = temp_df['pred'] * temp_df['weight']

    grouped = temp_df.groupby(['date_id', 'seconds_in_bucket'])
    group_sums = grouped['weighted_pred'].transform('sum')
    total_weights = grouped['weight'].transform('sum')

    weighted_means = group_sums / (total_weights + 1e-6)
    simple_means = grouped['pred'].transform('mean')
    weighted_means = np.where(total_weights < 1e-6, simple_means, weighted_means)

    return (temp_df['pred'] - weighted_means).values


def train_fold(X_train, y_train, X_val, y_val, val_df, stock_weights: dict, fold: int) -> dict:
    """
    Train XGBoost and LightGBM on one fold, ensemble, and post-process.

    Ensemble weights are computed as inverse-MAE: a model with lower validation
    error contributes proportionally more to the final prediction.
    """
    print(f"\n  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=CONFIG['early_stopping'],
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    xgb_preds = xgb_model.predict(X_val)
    xgb_mae = mean_absolute_error(y_val, xgb_preds)
    print(f"  XGBoost MAE: {xgb_mae:.5f} (best_iter: {xgb_model.best_iteration})")

    print(f"\n  Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(CONFIG['early_stopping']), lgb.log_evaluation(100)],
    )
    lgb_preds = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_preds)
    print(f"  LightGBM MAE: {lgb_mae:.5f} (best_iter: {lgb_model.best_iteration_})")

    # Inverse-MAE ensemble weighting
    w_xgb = 1 / (xgb_mae + 1e-6)
    w_lgb = 1 / (lgb_mae + 1e-6)
    w_total = w_xgb + w_lgb

    ensemble_preds = (w_xgb * xgb_preds + w_lgb * lgb_preds) / w_total
    ensemble_mae = mean_absolute_error(y_val, ensemble_preds)
    print(f"  Ensemble MAE: {ensemble_mae:.5f}")

    adjusted_preds = weighted_zero_sum_adjustment(ensemble_preds, val_df, stock_weights)
    adjusted_mae = mean_absolute_error(y_val, adjusted_preds)
    print(f"  Adjusted MAE: {adjusted_mae:.5f}")

    return {
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'xgb_preds': xgb_preds,
        'lgb_preds': lgb_preds,
        'ensemble_preds': ensemble_preds,
        'adjusted_preds': adjusted_preds,
        'xgb_mae': xgb_mae,
        'lgb_mae': lgb_mae,
        'ensemble_mae': ensemble_mae,
        'adjusted_mae': adjusted_mae,
        'xgb_weight': w_xgb / w_total,
        'lgb_weight': w_lgb / w_total,
    }
