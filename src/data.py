import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def reduce_mem_usage(df):
    """Downcast numeric columns to reduce memory footprint."""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory: {start_mem:.1f}MB → {end_mem:.1f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)')
    return df


def load_data(path: str, rapid_mode: bool = False, rapid_frac: float = 0.05) -> pd.DataFrame:
    """
    Load and optionally subsample the training CSV.

    Parameters
    ----------
    path        : Path to train.csv (local or Kaggle input path)
    rapid_mode  : If True, use only the first `rapid_frac` of dates for fast iteration
    rapid_frac  : Fraction of dates to keep in rapid mode (default 5%)
    """
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    df = reduce_mem_usage(df)

    if rapid_mode:
        n_dates = df['date_id'].nunique()
        cutoff = int(n_dates * rapid_frac)
        print(f"RAPID MODE: using first {cutoff} of {n_dates} dates.")
        df = df[df['date_id'] < cutoff].reset_index(drop=True)

    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date_id'].min()} – {df['date_id'].max()}")
    print(f"Stocks: {df['stock_id'].nunique()}")
    return df


def estimate_stock_weights(df: pd.DataFrame) -> tuple[dict, list, pd.DataFrame]:
    """
    Estimate index weights from matched_size (proxy for market cap).

    Returns
    -------
    weight_dict  : {stock_id: estimated_weight}
    lead_stocks  : list of top-10 stock_ids by weight
    stock_stats  : full per-stock DataFrame with estimated_weight column
    """
    print("Estimating stock weights...")

    stock_stats = df.groupby('stock_id').agg(
        matched_size=('matched_size', 'mean'),
        bid_size=('bid_size', 'mean'),
        ask_size=('ask_size', 'mean'),
        wap=('wap', 'mean'),
    )

    total = stock_stats['matched_size'].sum()
    stock_stats['estimated_weight'] = stock_stats['matched_size'] / total
    stock_stats['estimated_weight'] /= stock_stats['estimated_weight'].sum()  # normalise

    lead_stocks = stock_stats.nlargest(10, 'estimated_weight').index.tolist()
    weight_dict = stock_stats['estimated_weight'].to_dict()

    print(f"  Stocks: {len(weight_dict)}")
    print(f"  Lead stocks (top 10): {lead_stocks}")
    print(f"  Weight range: {stock_stats['estimated_weight'].min():.6f} – {stock_stats['estimated_weight'].max():.4f}")

    return weight_dict, lead_stocks, stock_stats


def create_stock_profiles(df: pd.DataFrame, stock_weights: dict, lead_stocks: list) -> pd.DataFrame:
    """
    Build per-stock feature profiles used downstream in feature engineering.
    """
    print("Creating stock profiles...")

    profiles = df.groupby('stock_id').agg(
        wap_mean=('wap', 'mean'),
        wap_std=('wap', 'std'),
        wap_min=('wap', 'min'),
        wap_max=('wap', 'max'),
        bid_price_mean=('bid_price', 'mean'),
        bid_price_std=('bid_price', 'std'),
        ask_price_mean=('ask_price', 'mean'),
        ask_price_std=('ask_price', 'std'),
        matched_size_mean=('matched_size', 'mean'),
        matched_size_median=('matched_size', 'median'),
        matched_size_std=('matched_size', 'std'),
        imbalance_size_mean=('imbalance_size', 'mean'),
        imbalance_size_median=('imbalance_size', 'median'),
        imbalance_size_std=('imbalance_size', 'std'),
        target_mean=('target', 'mean'),
        target_std=('target', 'std'),
        target_skew=('target', 'skew'),
        target_q05=('target', lambda x: x.quantile(0.05)),
        target_q95=('target', lambda x: x.quantile(0.95)),
        imbalance_flag_mean=('imbalance_buy_sell_flag', 'mean'),
        imbalance_flag_std=('imbalance_buy_sell_flag', 'std'),
    )

    profiles['estimated_weight'] = profiles.index.map(stock_weights)
    profiles['wap_volatility'] = profiles['wap_std'] / (profiles['wap_mean'] + 1e-6)
    profiles['size_ratio'] = profiles['matched_size_mean'] / (profiles['imbalance_size_mean'] + 1e-6)
    profiles['spread_mean'] = profiles['ask_price_mean'] - profiles['bid_price_mean']
    profiles['target_range'] = profiles['target_q95'] - profiles['target_q05']
    profiles['is_lead'] = profiles.index.isin(lead_stocks)

    print(f"  Profile shape: {profiles.shape}")
    return profiles


def cluster_stocks(profiles: pd.DataFrame, n_clusters: int = 9) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    KMeans clustering of stocks by behavioural profile.

    Returns updated profiles (with 'cluster' column), cluster summary, and fitted scaler.
    """
    print(f"Clustering stocks into {n_clusters} groups...")

    cluster_features = [
        'wap_mean', 'wap_volatility',
        'matched_size_mean', 'imbalance_size_mean',
        'target_std', 'target_range',
        'estimated_weight', 'spread_mean',
    ]

    X = profiles[cluster_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    profiles = profiles.copy()
    profiles['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_summary = profiles.groupby('cluster').agg(
        wap_mean=('wap_mean', 'mean'),
        wap_volatility=('wap_volatility', 'mean'),
        target_std=('target_std', 'mean'),
        weight_mean=('estimated_weight', 'mean'),
        weight_sum=('estimated_weight', 'sum'),
        count=('estimated_weight', 'count'),
        lead_stocks=('is_lead', 'sum'),
    )

    return profiles, cluster_summary, scaler
