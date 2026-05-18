import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

from src.config import COLORS


def plot_target_analysis(df, stock_weights):

    sns.set_context("notebook", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Part 1: Target Analysis', fontsize=20, fontweight='bold', y=1.02)

    ax = axes[0, 0]
    target_clean = df['target'].dropna()
    ax.hist(target_clean, bins=100, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=target_clean.mean(), color='green', linestyle='--', label=f'Mean: {target_clean.mean():.2f}')
    ax.set_xlabel('Target (basis points)')
    ax.set_ylabel('Frequency')
    ax.set_title('Target Distribution')
    ax.legend()

    ax = axes[0, 1]
    target_by_time = df.groupby('seconds_in_bucket')['target'].agg(['mean', 'std'])
    ax.fill_between(target_by_time.index,
                    target_by_time['mean'] - target_by_time['std'],
                    target_by_time['mean'] + target_by_time['std'],
                    alpha=0.3, color=COLORS['primary'])
    ax.plot(target_by_time.index, target_by_time['mean'], color=COLORS['primary'], linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Seconds in Bucket')
    ax.set_ylabel('Target Mean ± Std')
    ax.set_title('2. Target by Time (Intraday)')

    ax = axes[1, 0]
    daily_stats = df.groupby('date_id')['target'].agg(['mean', 'std'])
    ax.fill_between(daily_stats.index, daily_stats['mean'] - daily_stats['std'],
                    daily_stats['mean'] + daily_stats['std'], alpha=0.3, color=COLORS['secondary'])
    ax.plot(daily_stats.index, daily_stats['mean'], color=COLORS['secondary'], linewidth=1)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date ID')
    ax.set_ylabel('Target Mean ± Std')
    ax.set_title('3. Daily Target Patterns')

    ax = axes[1, 1]
    df_temp = df.copy()
    df_temp['stock_weight'] = df_temp['stock_id'].map(stock_weights)
    weight_percentiles = df_temp['stock_weight'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    weight_percentiles = np.unique(weight_percentiles)
    if len(weight_percentiles) >= 2:
        df_temp['weight_bin'] = pd.cut(df_temp['stock_weight'],
                                       bins=weight_percentiles,
                                       labels=[f'Q{i+1}' for i in range(len(weight_percentiles)-1)],
                                       include_lowest=True)
        target_by_weight = df_temp.groupby('weight_bin', observed=True)['target'].std()
        ax.bar(range(len(target_by_weight)), target_by_weight.values, color=COLORS['accent'], alpha=0.7)
        ax.set_xticks(range(len(target_by_weight)))
        ax.set_xticklabels(target_by_weight.index)
    ax.set_ylabel('Target Std Dev')
    ax.set_xlabel('Stock Weight Quantile (Q1=Smallest, Q5=Largest)')
    ax.set_title('4. Target Volatility by Stock Weight')
    del df_temp

    plt.tight_layout()
    plt.show()


def plot_microstructure_analysis(df):

    sns.set_context("notebook", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle('Part 2: Market Microstructure Analysis', fontsize=20, fontweight='bold', y=1.02)

    ax = axes[0]
    flag_counts = df['imbalance_buy_sell_flag'].value_counts().sort_index()
    flag_indices = [-1, 0, 1]
    flag_values = [flag_counts.get(i, 0) for i in flag_indices]
    ax.bar(flag_indices, flag_values, color=['red', 'gray', 'green'], alpha=0.7)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['Sell (-1)', 'Neutral (0)', 'Buy (+1)'])
    ax.set_ylabel('Count')
    ax.set_title('1. Imbalance Direction Distribution')

    ax = axes[1]
    df_imb = df.copy()
    df_imb['norm_imbalance'] = df_imb['imbalance_size'] / (df_imb['imbalance_size'] + df_imb['matched_size'] + 1e-6)
    imb_by_sec = df_imb.groupby(['seconds_in_bucket', 'imbalance_buy_sell_flag'])['norm_imbalance'].mean().unstack()
    if 1 in imb_by_sec.columns:
        ax.plot(imb_by_sec.index, imb_by_sec[1], color='green', label='Buy Imbalance', linewidth=2)
    if -1 in imb_by_sec.columns:
        ax.plot(imb_by_sec.index, imb_by_sec[-1], color='red', label='Sell Imbalance', linewidth=2)
    ax.set_xlabel('Seconds in Bucket')
    ax.set_ylabel('Avg Normalized Imbalance Ratio')
    ax.set_title('2. Imbalance Magnitude vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    spread = df['ask_price'] - df['bid_price']
    spread_clean = spread[spread > 0].dropna()
    ax.hist(spread_clean, bins=100, color=COLORS['success'], alpha=0.7)
    ax.set_xlabel('Spread (Ask - Bid)')
    ax.set_ylabel('Frequency')
    ax.set_title('3. Spread Distribution (Positive Only)')
    ax.set_xlim(0, spread_clean.quantile(0.99))

    plt.tight_layout()
    plt.show()


def plot_cluster_analysis(profiles, cluster_summary):

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Stock Cluster Analysis', fontsize=18, fontweight='bold')

    n_clusters = profiles['cluster'].nunique()
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    ax = axes[0, 0]
    cluster_sizes = profiles['cluster'].value_counts().sort_index()
    bars = ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Stocks')
    ax.set_title('Stocks per Cluster')
    for bar, val in zip(bars, cluster_sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[0, 1]
    for cluster in range(n_clusters):
        mask = profiles['cluster'] == cluster
        ax.scatter(profiles.loc[mask, 'wap_volatility'],
                   profiles.loc[mask, 'target_std'],
                   c=[colors[cluster]], label=f'C{cluster}', s=80, alpha=0.7)
    ax.set_xlabel('WAP Volatility')
    ax.set_ylabel('Target Std Dev')
    ax.set_title('Volatility Characteristics by Cluster')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

    ax = axes[0, 2]
    weight_by_cluster = profiles.groupby('cluster')['estimated_weight'].sum().sort_values(ascending=False)
    ax.bar(range(len(weight_by_cluster)), weight_by_cluster.values,
           color=[colors[i] for i in weight_by_cluster.index])
    ax.set_xticks(range(len(weight_by_cluster)))
    ax.set_xticklabels([f'C{i}' for i in weight_by_cluster.index])
    ax.set_ylabel('Total Estimated Weight')
    ax.set_title('Weight Concentration by Cluster')

    ax = axes[1, 0]
    char_cols = ['wap_volatility', 'target_std', 'estimated_weight', 'matched_size_mean', 'spread_mean']
    cluster_chars = profiles.groupby('cluster')[char_cols].mean()
    cluster_chars_norm = (cluster_chars - cluster_chars.min()) / (cluster_chars.max() - cluster_chars.min() + 1e-6)
    im = ax.imshow(cluster_chars_norm.T, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax.set_yticks(range(len(char_cols)))
    ax.set_yticklabels(char_cols)
    ax.set_title('Normalized Cluster Characteristics')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    lead_by_cluster = profiles[profiles['is_lead']].groupby('cluster').size()
    all_clusters = pd.Series(0, index=range(n_clusters))
    lead_by_cluster = all_clusters.add(lead_by_cluster, fill_value=0)
    ax.bar(lead_by_cluster.index, lead_by_cluster.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Lead Stocks')
    ax.set_title('Lead Stock Distribution')

    ax = axes[1, 2]
    target_by_cluster = profiles.groupby('cluster')['target_std'].mean().sort_values(ascending=False)
    ax.barh(range(len(target_by_cluster)), target_by_cluster.values,
            color=[colors[i] for i in target_by_cluster.index])
    ax.set_yticks(range(len(target_by_cluster)))
    ax.set_yticklabels([f'Cluster {i}' for i in target_by_cluster.index])
    ax.set_xlabel('Average Target Std Dev')
    ax.set_title('Cluster Volatility Ranking')

    plt.tight_layout()
    plt.show()


def plot_model_comparison(fold_results):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    n_folds = len(fold_results)
    folds = list(range(1, n_folds + 1))

    xgb_maes = [r['xgb_mae'] for r in fold_results]
    lgb_maes = [r['lgb_mae'] for r in fold_results]
    ens_maes = [r['ensemble_mae'] for r in fold_results]
    adj_maes = [r['adjusted_mae'] for r in fold_results]

    ax = axes[0, 0]
    x = np.arange(n_folds)
    width = 0.2
    ax.bar(x - 1.5*width, xgb_maes, width, label='XGBoost', color=COLORS['primary'])
    ax.bar(x - 0.5*width, lgb_maes, width, label='LightGBM', color=COLORS['secondary'])
    ax.bar(x + 0.5*width, ens_maes, width, label='Ensemble', color=COLORS['accent'])
    ax.bar(x + 1.5*width, adj_maes, width, label='Adjusted', color=COLORS['success'])
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Model and Fold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 1]
    models = ['XGBoost', 'LightGBM', 'Ensemble', 'Adjusted']
    avg_maes = [np.mean(xgb_maes), np.mean(lgb_maes), np.mean(ens_maes), np.mean(adj_maes)]
    std_maes = [np.std(xgb_maes), np.std(lgb_maes), np.std(ens_maes), np.std(adj_maes)]
    bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
    bars = ax.bar(models, avg_maes, yerr=std_maes, capsize=5, color=bar_colors, alpha=0.7)
    ax.set_ylabel('Average MAE')
    ax.set_title('Average Performance (±std)')
    for bar, mae in zip(bars, avg_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 2]
    periods = ['Early', 'Middle', 'Late']
    period_avgs = [np.mean([r[f'{p}_mae'] for r in fold_results]) for p in ['early', 'middle', 'late']]
    bars = ax.bar(periods, period_avgs, color=[COLORS['success'], COLORS['accent'], COLORS['primary']])
    ax.set_ylabel('Average MAE')
    ax.set_title('MAE by Time Period')
    for bar, mae in zip(bars, period_avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    improvements = [e - a for e, a in zip(ens_maes, adj_maes)]
    colors_imp = [COLORS['success'] if imp > 0 else COLORS['accent'] for imp in improvements]
    ax.bar(folds, improvements, color=colors_imp, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE Improvement')
    ax.set_title('Post-Processing Improvement')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    xgb_weights = [r['xgb_weight'] for r in fold_results]
    lgb_weights = [r['lgb_weight'] for r in fold_results]
    ax.bar(folds, xgb_weights, label='XGBoost', color=COLORS['primary'])
    ax.bar(folds, lgb_weights, bottom=xgb_weights, label='LightGBM', color=COLORS['secondary'])
    ax.set_xlabel('Fold')
    ax.set_ylabel('Weight')
    ax.set_title('Ensemble Weights by Fold')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"FINAL RESULTS SUMMARY\n{'='*40}\n\n"
        f"XGBoost:  {np.mean(xgb_maes):.5f} ± {np.std(xgb_maes):.5f}\n"
        f"LightGBM: {np.mean(lgb_maes):.5f} ± {np.std(lgb_maes):.5f}\n"
        f"Ensemble: {np.mean(ens_maes):.5f} ± {np.std(ens_maes):.5f}\n"
        f"Adjusted: {np.mean(adj_maes):.5f} ± {np.std(adj_maes):.5f}\n\n"
        f"Post-Processing Gain: {np.mean(improvements):.5f}\n\n"
        f"Period Breakdown:\n"
        f"  Early  (0-200s):   {period_avgs[0]:.5f}\n"
        f"  Middle (200-400s): {period_avgs[1]:.5f}\n"
        f"  Late   (400-600s): {period_avgs[2]:.5f}\n\n"
        f"Best Fold:  {folds[np.argmin(adj_maes)]} ({min(adj_maes):.5f})\n"
        f"Worst Fold: {folds[np.argmax(adj_maes)]} ({max(adj_maes):.5f})"
    )
    ax.text(0.05, 0.95, summary, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def plot_detailed_fold_analysis(fold_results, stock_profiles, lead_stocks: list, fold_idx: int = -1):
    """
    Detailed per-fold analysis: predictions, errors, feature importances, cluster breakdown.

    Parameters
    ----------
    lead_stocks : list of stock_ids identified as index leaders (passed explicitly
                  to avoid a global variable dependency)
    """
    results = fold_results[fold_idx]
    val_df = results['val_df']
    y_true = results['y_val'].values
    preds = results['adjusted_preds']
    errors = np.abs(y_true - preds)

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    actual_fold = len(fold_results) if fold_idx == -1 else fold_idx + 1
    fig.suptitle(f'Fold {actual_fold} Detailed Analysis', fontsize=16, fontweight='bold')

    sample_idx = np.random.choice(len(y_true), min(5000, len(y_true)), replace=False)

    ax = axes[0, 0]
    ax.scatter(y_true[sample_idx], preds[sample_idx], alpha=0.3, s=10, c=COLORS['primary'])
    lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predictions vs Actual')

    ax = axes[0, 1]
    ax.hist(errors, bins=100, color=COLORS['primary'], alpha=0.7)
    ax.axvline(x=np.median(errors), color='red', linestyle='--', label=f'Median: {np.median(errors):.2f}')
    ax.axvline(x=np.mean(errors), color='green', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()

    ax = axes[0, 2]
    time_mae, time_labels = [], []
    for t in range(0, 600, 10):
        mask = (val_df['seconds_in_bucket'] >= t) & (val_df['seconds_in_bucket'] < t + 10)
        if mask.sum() > 0:
            time_mae.append(mean_absolute_error(y_true[mask], preds[mask]))
            time_labels.append(t)
    ax.plot(time_labels, time_mae, color=COLORS['primary'], linewidth=2)
    ax.axvspan(200, 400, alpha=0.2, color='red', label='Middle Period')
    ax.set_xlabel('Seconds in Bucket')
    ax.set_ylabel('MAE')
    ax.set_title('MAE Throughout Auction')
    ax.legend()

    ax = axes[0, 3]
    residuals = y_true - preds
    ax.scatter(preds[sample_idx], residuals[sample_idx], alpha=0.3, s=10, c=COLORS['secondary'])
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predictions')

    ax = axes[1, 0]
    val_df_copy = val_df.copy()
    val_df_copy['error'] = errors
    val_df_copy['cluster'] = val_df_copy['stock_id'].map(stock_profiles['cluster'].to_dict())
    cluster_mae = val_df_copy.groupby('cluster')['error'].mean().sort_values(ascending=False)
    colors_cluster = plt.cm.tab20(np.linspace(0, 1, len(cluster_mae)))
    ax.barh(range(len(cluster_mae)), cluster_mae.values, color=colors_cluster)
    ax.set_yticks(range(len(cluster_mae)))
    ax.set_yticklabels([f'Cluster {c}' for c in cluster_mae.index])
    ax.set_xlabel('MAE')
    ax.set_title('MAE by Stock Cluster')
    ax.invert_yaxis()

    ax = axes[1, 1]
    ax.hist(y_true, bins=100, alpha=0.5, label='Actual', color=COLORS['primary'])
    ax.hist(preds, bins=100, alpha=0.5, label='Predicted', color=COLORS['accent'])
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution Comparison')
    ax.legend()

    ax = axes[1, 2]
    ts_means = val_df.copy()
    ts_means['pred'] = preds
    ts_means_grouped = ts_means.groupby(['date_id', 'seconds_in_bucket'])['pred'].mean()
    ax.hist(ts_means_grouped, bins=50, color=COLORS['secondary'], alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Mean Prediction per Timestamp')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Zero-Sum Check (std={ts_means_grouped.std():.4f})')

    ax = axes[1, 3]
    stock_mae = val_df_copy.groupby('stock_id')['error'].mean().sort_values(ascending=False).head(20)
    ax.barh(range(len(stock_mae)), stock_mae.values, color=COLORS['success'])
    ax.set_yticks(range(len(stock_mae)))
    ax.set_yticklabels([f'Stock {s}' for s in stock_mae.index], fontsize=8)
    ax.set_xlabel('MAE')
    ax.set_title('Top 20 Hardest Stocks')
    ax.invert_yaxis()

    ax = axes[2, 0]
    importance = results['xgb_model'].feature_importances_
    features = results['xgb_model'].feature_names_in_
    top_idx = np.argsort(importance)[-15:]
    ax.barh(range(15), importance[top_idx], color=COLORS['primary'])
    ax.set_yticks(range(15))
    ax.set_yticklabels(features[top_idx], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features (XGBoost)')

    ax = axes[2, 1]
    importance = results['lgb_model'].feature_importances_
    features = results['lgb_model'].feature_name_
    top_idx = np.argsort(importance)[-15:]
    ax.barh(range(15), importance[top_idx], color=COLORS['secondary'])
    ax.set_yticks(range(15))
    ax.set_yticklabels([features[i] for i in top_idx], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features (LightGBM)')

    ax = axes[2, 2]
    val_df_copy['is_lead'] = val_df_copy['stock_id'].isin(lead_stocks)
    lead_mae_val = val_df_copy[val_df_copy['is_lead']]['error'].mean()
    nonlead_mae_val = val_df_copy[~val_df_copy['is_lead']]['error'].mean()
    ax.bar(['Lead Stocks', 'Other Stocks'], [lead_mae_val, nonlead_mae_val],
           color=[COLORS['accent'], COLORS['primary']], alpha=0.7)
    ax.set_ylabel('MAE')
    ax.set_title('Lead vs Non-Lead Stocks')
    for i, mae in enumerate([lead_mae_val, nonlead_mae_val]):
        ax.text(i, mae + 0.02, f'{mae:.4f}', ha='center', fontsize=11, fontweight='bold')

    ax = axes[2, 3]
    ax.axis('off')
    summary = (
        f"FOLD {actual_fold} STATISTICS\n{'='*35}\n\n"
        f"Sample Size: {len(y_true):,}\n\n"
        f"MAE Scores:\n"
        f"  XGBoost:  {results['xgb_mae']:.5f}\n"
        f"  LightGBM: {results['lgb_mae']:.5f}\n"
        f"  Ensemble: {results['ensemble_mae']:.5f}\n"
        f"  Adjusted: {results['adjusted_mae']:.5f}\n\n"
        f"Prediction Range:\n"
        f"  Actual: [{y_true.min():.1f}, {y_true.max():.1f}]\n"
        f"  Pred:   [{preds.min():.1f}, {preds.max():.1f}]\n\n"
        f"Error Stats:\n"
        f"  Mean:   {errors.mean():.4f}\n"
        f"  Median: {np.median(errors):.4f}\n"
        f"  Max:    {errors.max():.2f}\n\n"
        f"Coverage:\n"
        f"  ±5 bps:  {(errors <= 5).mean():.1%}\n"
        f"  ±10 bps: {(errors <= 10).mean():.1%}"
    )
    ax.text(0.05, 0.95, summary, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.show()
