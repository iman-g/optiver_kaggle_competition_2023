# Optiver Trading at the Close - Stock Price Prediction

Predicting stock price movements in the last 10 minutes before NASDAQ closing auction.

## What's This About?

The target is simple: predict how much a stock will move relative to a synthetic index in the 60 seconds after auction close. The catch? You only get order book data during the final 10 minutes of trading when things get chaotic.

## My Approach

### 1. Data Exploration

Started with 5.2M rows across 481 trading days and 200 stocks.

**Key findings from EDA:**
- Target is centered around zero (mean: -0.05 bps) with fat tails ranging from -400 to +400
- Middle period (200-400s) has the highest volatility - this is where predictions struggle most
- Smaller stocks (low weight) are much harder to predict than large caps
- Buy/Sell imbalance is roughly balanced, but imbalance magnitude drops sharply toward auction close

### 2. Stock Weight Estimation

Instead of using hardcoded index weights, I estimated them from the data using `matched_size` as a proxy for market cap:

```
Top 10 Lead Stocks: [112, 45, 168, 191, 41, 175, 179, 84, 142, 95]
Weight range: 0.007% to 9.96%
```

### 3. Stock Clustering

Used KMeans to group stocks by behavior. Elbow method suggested 9 clusters.

**Cluster characteristics:**
- Cluster 4 & 7: High-weight stocks, lower volatility, contains most lead stocks
- Cluster 5: Tiny stocks with extreme volatility (target std: 20!)
- Cluster 2: Just 2 mega-cap stocks making up 19% of index weight

### 4. Feature Engineering (115 features)

**Basic:** spread, volume, liquidity imbalance, price comparisons

**Time-based:** lag features (1,2,3,5,10 periods), returns, diffs

**Cross-sectional:** performance vs market WAP, vs weighted index, percentile ranks

**Clever stuff from winning solutions:**
- Inferred price from tick size (reverse-engineering actual stock prices)
- Revealed targets (yesterday's target as today's feature)
- Weighted index returns

### 5. Model

**Architecture:** XGBoost + LightGBM ensemble with inverse-MAE weighting

**CV:** 5-fold purged time series split with 5-day gap to prevent leakage

**Post-processing:** Weighted zero-sum adjustment per timestamp

## Results

| Model | MAE | Std |
|-------|-----|-----|
| XGBoost | 6.463 | 0.521 |
| LightGBM | 6.460 | 0.516 |
| Ensemble | 6.460 | 0.518 |
| **Adjusted** | **6.455** | 0.514 |

**By time period:**
- Early (0-200s): 6.31
- Middle (200-400s): 7.28 ← hardest
- Late (400-600s): 5.54 ← easiest

**Best fold:** 5.86 MAE

### What Worked

1. **LightGBM slightly beats XGBoost** - consistent across all folds
2. **Zero-sum post-processing helps** - small but free improvement
3. **Lead stocks are easier** - 4.27 MAE vs 5.94 for others
4. **Time features matter** - `seconds_in_bucket` is top feature in LightGBM

### What's Still Hard

1. **Middle period volatility** - MAE jumps from 6.3 to 7.3 in the 200-400s window
2. **Small/volatile stocks** - Cluster 5 and 8 have 15-20 MAE
3. **Extreme values** - actual range is [-225, 388] but predictions stay in [-24, 32]

## Top Features

**XGBoost:** size_imbalance, liquidity_imbalance, reference_price_wap_imb

**LightGBM:** seconds_in_bucket, market_imbalance, market_flag, stock_wap_volatility

## Files

- `optiver_notebook.ipynb` - Full training pipeline
- `README.md` - You're reading it

## What I'd Try Next

1. Neural network for the middle period specifically
2. Stock-specific models for the volatile clusters
3. More aggressive feature selection (drop correlated features)
4. Quantile regression to capture the tails better

## Acknowledgments

Ideas borrowed from:
- [lognorm's winning solution](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion) - index weights, inferred price, revealed targets
- Various public notebooks for triplet imbalance and baseline features

---

*Competition: [Optiver - Trading at the Close (Kaggle 2023-2024)](https://www.kaggle.com/competitions/optiver-trading-at-the-close)*
