import numpy as np
import pandas as pd
from src.data import estimate_stock_weights, create_stock_profiles, cluster_stocks
from src.features import create_features, create_revealed_targets
from src.model import PurgedGroupTimeSeriesSplit, prepare_train_val, weighted_zero_sum_adjustment


def make_fake_data(n_stocks=10, n_dates=30, n_seconds=10):
    """Build a minimal dataframe that mimics the Optiver schema."""
    rows = []
    for date_id in range(n_dates):
        for stock_id in range(n_stocks):
            for seconds in range(0, 600, 600 // n_seconds):
                rows.append({
                    'date_id': date_id,
                    'stock_id': stock_id,
                    'seconds_in_bucket': seconds,
                    'bid_price': np.random.uniform(90, 110),
                    'ask_price': np.random.uniform(90, 110),
                    'bid_size': np.random.uniform(100, 1000),
                    'ask_size': np.random.uniform(100, 1000),
                    'wap': np.random.uniform(90, 110),
                    'reference_price': np.random.uniform(90, 110),
                    'far_price': np.random.uniform(90, 110),
                    'near_price': np.random.uniform(90, 110),
                    'imbalance_size': np.random.uniform(0, 500),
                    'matched_size': np.random.uniform(100, 1000),
                    'imbalance_buy_sell_flag': np.random.choice([-1, 0, 1]),
                    'target': np.random.normal(0, 5),
                    'row_id': f"{date_id}_{stock_id}_{seconds}",
                    'time_id': date_id * 1000 + seconds,
                    'currently_scored': True,
                })
    return pd.DataFrame(rows)


def test_stock_weights():
    df = make_fake_data()
    weights, lead_stocks, stats = estimate_stock_weights(df)
    assert len(weights) == 10
    assert len(lead_stocks) == 10
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"
    print("PASS: test_stock_weights")


def test_purged_cv_no_leakage():
    df = make_fake_data(n_dates=30)
    splitter = PurgedGroupTimeSeriesSplit(n_splits=3, purge_gap=2)
    date_ids = df['date_id']
    for train_idx, val_idx in splitter.split(df, date_ids):
        train_dates = df[train_idx]['date_id'].unique()
        val_dates = df[val_idx]['date_id'].unique()
        assert max(train_dates) < min(val_dates), "Train dates must all precede val dates"
    print("PASS: test_purged_cv_no_leakage")


def test_zero_sum_adjustment():
    df = make_fake_data()
    weights, _, _ = estimate_stock_weights(df)
    fake_preds = np.random.normal(0, 5, len(df))
    adjusted = weighted_zero_sum_adjustment(fake_preds, df, weights)

    # After adjustment, weighted sum per timestamp should be ~0
    df['adjusted'] = adjusted
    df['weight'] = df['stock_id'].map(weights)
    df['weighted_adj'] = df['adjusted'] * df['weight']
    grouped = df.groupby(['date_id', 'seconds_in_bucket'])['weighted_adj'].sum()
    assert grouped.abs().max() < 1e-4, f"Weighted sum not zero: max={grouped.abs().max()}"
    print("PASS: test_zero_sum_adjustment")


def test_feature_engineering_runs():
    df = make_fake_data()
    weights, lead_stocks, _ = estimate_stock_weights(df)
    profiles = create_stock_profiles(df, weights, lead_stocks)
    profiles, _, _ = cluster_stocks(profiles, n_clusters=3)
    revealed = create_revealed_targets(df)
    df_feat = create_features(df, profiles, weights, lead_stocks, revealed)
    assert df_feat.shape[1] > 20, "Expected many features"
    assert df_feat.isnull().sum().sum() == 0, "No nulls expected after fillna"
    print(f"PASS: test_feature_engineering_runs — {df_feat.shape[1]} features created")


if __name__ == "__main__":
    test_stock_weights()
    test_purged_cv_no_leakage()
    test_zero_sum_adjustment()
    test_feature_engineering_runs()
    print("\nAll tests passed.")