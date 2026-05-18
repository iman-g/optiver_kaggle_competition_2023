import pandas as pd
import numpy as np
import warnings
from itertools import combinations


def impute_missing(df):
    df = df.copy()
    
    cols = ['imbalance_size', 'reference_price', 'matched_size', 'wap', 'bid_price', 'ask_price']
    df[cols] = df.groupby(['stock_id', 'date_id'])[cols].ffill().bfill()
    
    df['far_price_null'] = df['far_price'].isna().astype(np.int8)
    df['near_price_null'] = df['near_price'].isna().astype(np.int8)
    df['far_price'] = df['far_price'].fillna(df['reference_price'])
    df['near_price'] = df['near_price'].fillna(df['reference_price'])
    
    return df


def create_revealed_targets(df):

    print("Creating revealed targets...")
    
    revealed = df[['date_id', 'seconds_in_bucket', 'stock_id', 'target']].copy()
    revealed = revealed.rename(columns={'target': 'revealed_target'})
    revealed['date_id'] = revealed['date_id'] + 1
    
    return revealed



def create_features(df, stock_profiles, stock_weights, lead_stocks, revealed_targets=None):
    print("Creating features...")
    
    df = impute_missing(df)
    df = df.sort_values(['stock_id', 'date_id', 'seconds_in_bucket']).reset_index(drop=True)
    
    # ================================================================
    # BASIC FEATURES
    # ================================================================
    print("  Basic features...")
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
    df['spread'] = df['ask_price'] - df['bid_price']
    df['spread_pct'] = df['spread'] / (df['wap'] + 1e-6)
    df['volume'] = df['bid_size'] + df['ask_size']
    
    # Imbalance features
    df['liquidity_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['volume'] + 1e-6)
    df['size_imbalance'] = df['bid_size'] / (df['ask_size'] + 1e-6)
    df['matched_imbalance'] = (df['imbalance_size'] - df['matched_size']) / (df['matched_size'] + df['imbalance_size'] + 1e-6)
    df['signed_imbalance'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']
    
    # Price comparisons
    prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    for p1, p2 in combinations(prices, 2):
        df[f'{p1}_{p2}_imb'] = (df[p1] - df[p2]) / (df[p1] + df[p2] + 1e-6)
    
    # Time features
    df['seconds'] = df['seconds_in_bucket'] % 60
    df['minute'] = df['seconds_in_bucket'] // 60
    df['time_pct'] = df['seconds_in_bucket'] / 600
    df['time_to_close'] = 600 - df['seconds_in_bucket']
    df['is_early'] = (df['seconds_in_bucket'] < 180).astype(np.int8)
    df['is_late'] = (df['seconds_in_bucket'] >= 420).astype(np.int8)
    
    # ================================================================
    # STOCK WEIGHT FEATURES
    # ================================================================
    print("  Stock weight features...")
    df['stock_weight'] = df['stock_id'].map(stock_weights).fillna(0).astype(np.float32)
    df['is_lead_stock'] = df['stock_id'].isin(lead_stocks).astype(np.int8)
    
    # ================================================================
    # SHIFT/LAG FEATURES
    # ================================================================
    print("  Lag features...")
    g = df.groupby(['stock_id', 'date_id'])
    
    for window in [1, 2, 3, 5, 10]:
        df[f'wap_shift_{window}'] = g['wap'].shift(window)
        df[f'imbalance_shift_{window}'] = g['imbalance_size'].shift(window)
        df[f'flag_shift_{window}'] = g['imbalance_buy_sell_flag'].shift(window)
        
        df[f'wap_ret_{window}'] = g['wap'].pct_change(window)
        df[f'imbalance_ret_{window}'] = g['imbalance_size'].pct_change(window)
        
        df[f'spread_diff_{window}'] = g['spread'].diff(window)
        df[f'volume_diff_{window}'] = g['volume'].diff(window)
    
    # ================================================================
    # CROSS-SECTIONAL FEATURES (MARKET-LEVEL)
    # ================================================================
    print("  Cross-sectional features...")
    ts = df.groupby(['date_id', 'seconds_in_bucket'])
    
    # Weighted index WAP
    df['weighted_wap'] = df['wap'] * df['stock_weight']
    df['index_wap'] = df.groupby(['date_id', 'seconds_in_bucket'])['weighted_wap'].transform('sum')
    
    # Simple market averages
    df['market_wap'] = ts['wap'].transform('mean')
    df['market_imbalance'] = ts['signed_imbalance'].transform('mean')
    df['market_flag'] = ts['imbalance_buy_sell_flag'].transform('mean')
    
    # Performance vs market
    df['perf_vs_market'] = 10000 * (df['wap'] - df['market_wap'])
    df['perf_vs_index'] = 10000 * (df['wap'] - df['index_wap'])
    
    # Cross-sectional ranks
    df['wap_rank'] = ts['wap'].rank(pct=True)
    df['imbalance_rank'] = ts['signed_imbalance'].rank(pct=True)
    
    # Total imbalance normalization
    total_imb = ts['imbalance_size'].transform('sum')
    df['imbalance_share'] = df['signed_imbalance'] / (total_imb + 1e-6)
    
    # ================================================================
    # INDEX RETURNS (WEIGHTED)
    # ================================================================
    print("  Index return features...")
    for window in [1, 2, 3, 5]:
        wap_change = df[f'wap_ret_{window}'].fillna(0)
        df[f'weighted_change_{window}'] = df['stock_weight'] * (wap_change + 1)
        df[f'index_ret_{window}'] = df.groupby(['date_id', 'seconds_in_bucket'])[f'weighted_change_{window}'].transform('sum')
        
        # Stock return vs index return
        stock_ret = df['wap'] / (df['wap'] - wap_change + 1e-6) - 1
        df[f'rel_perf_{window}'] = 10000 * (stock_ret - df[f'index_ret_{window}'])
        
        del df[f'weighted_change_{window}']
    
    # ================================================================
    # INFERRED PRICE FROM TICK SIZE
    # ================================================================
    print("  Inferred price features...")
    df['price_move'] = g['bid_price'].diff().abs()
    df.loc[df['price_move'] == 0, 'price_move'] = np.nan
    
    df['tick_size'] = g['price_move'].transform(
        lambda x: x.rolling(55, min_periods=1).min()
    )
    df['inferred_price'] = 0.01 / (df['tick_size'] + 1e-9)
    df['inferred_price'] = df['inferred_price'].clip(upper=1000)
    
    # Inferred volumes
    df['inferred_bid_vol'] = df['bid_size'] / (df['inferred_price'] + 1e-6)
    df['inferred_ask_vol'] = df['ask_size'] / (df['inferred_price'] + 1e-6)
    df['inferred_imb_vol'] = df['imbalance_size'] / (df['inferred_price'] + 1e-6)
    
    del df['price_move']
    
    # ================================================================
    # VOLATILITY FEATURES
    # ================================================================
    print("  Volatility features...")
    df['wap_vol_10'] = g['wap_ret_1'].transform(
        lambda x: x.rolling(10, min_periods=1).std()
    )
    
    # ================================================================
    # STOCK PROFILE FEATURES
    # ================================================================
    print("  Stock profile features...")
    profile_cols = ['cluster', 'wap_volatility', 'target_std', 'target_mean', 'estimated_weight']
    for col in profile_cols:
        if col in stock_profiles.columns:
            df[f'stock_{col}'] = df['stock_id'].map(stock_profiles[col].to_dict())
    
    # ================================================================
    # REVEALED TARGET FEATURES
    # ================================================================
    # NEW (fixed) - create a fresh groupby AFTER the merge
    if revealed_targets is not None:
        print("  Revealed target features...")
        df = df.merge(
            revealed_targets[['date_id', 'seconds_in_bucket', 'stock_id', 'revealed_target']],
            on=['date_id', 'seconds_in_bucket', 'stock_id'],
            how='left'
        )
        
        # Interaction feature
        df['revealed_x_flag'] = df['revealed_target'] * df['imbalance_buy_sell_flag']
        
        g_revealed = df.groupby(['stock_id', 'date_id'])
        df['revealed_roll'] = g_revealed['revealed_target'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
    
    # ================================================================
    # CLEANUP
    # ================================================================
    print("  Cleaning up...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    drop_cols = ['weighted_wap', 'tick_size']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    print(f"  Total features: {df.shape[1]}")
    return df