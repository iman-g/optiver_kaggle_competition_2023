# Optiver Trading at the Close (Stock Price Prediction)

Predicting stock price movements in the last 10 minutes before the NASDAQ closing auction.

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/optiver-trading-at-the-close)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](requirements.txt)

---

## What's This About?

Predict how much each of 200 NASDAQ stocks will move relative to a synthetic index
in the 60 seconds after auction close, using only order book data from the final
10 minutes of trading. 5.2M rows, 481 trading days.

**The core challenge:** the target has fat tails (±400 bps actual range) but
standard MAE-optimized models collapse predictions to a narrow band near zero,
exactly where they're least useful for risk-aware decision making.

---

## Repository Structure

```
src/
├── config.py          # hyperparameters and model params
├── data.py            # loading, memory reduction, stock weights, clustering
├── features.py        # 115-feature engineering pipeline
├── model.py           # PurgedGroupTimeSeriesSplit, ensemble, zero-sum adjustment
├── evaluation.py      # CV summary, JSON result persistence
├── visualization.py   # all plot functions
└── optimize.py        # feature selection, DART, quantile regression

scripts/
└── train.py           # CLI pipeline: --data, --rapid, --no-optimize

tests/
└── test_pipeline.py   # logic tests with synthetic data (no Kaggle data needed)

results/
└── cv_results.json    # persisted CV metrics
```

**Run locally:**
```bash
pip install -r requirements.txt
python -m scripts.train --data /path/to/train.csv --rapid   # smoke test
python -m scripts.train --data /path/to/train.csv           # full run
```

---

## Key Design Decisions

### Purged Time-Series Cross-Validation

Standard k-fold CV shuffles data randomly. That's wrong for time-series, the
validation set ends up containing data from *before* the training set, so the model
has effectively seen the future. For stock data this is especially dangerous because
consecutive days are highly correlated.

Purged CV keeps strict chronological order: training always precedes validation.
The 5-day purge gap removes days immediately before the validation window from
training. Without it, lag features computed on day N leak information about day N+5.
The gap size equals the longest lag feature, this is not optional.

A test in `tests/test_pipeline.py` asserts that the maximum training date is always
strictly less than the minimum validation date across all folds, a concrete
guarantee of no leakage, not just an assumption.

### Weighted Zero-Sum Adjustment

The target measures each stock's movement relative to a weighted index. That index
is a weighted average of all stocks. Therefore the weighted sum of all targets per
timestamp must equal zero by definition, if some stocks go up relative to the
index, others must go down by the same weighted amount.

Raw model predictions violate this constraint. Post-processing subtracts the
weighted mean prediction per timestamp, enforcing the mathematical constraint.
This is a free performance gain, not a model choice, but a hard rule baked
into the problem definition.

### Inverse-MAE Ensemble Weighting

Rather than a fixed 50/50 split, each model receives weight proportional to
1/MAE computed on the validation fold. A model with MAE 6.40 contributes more
than one with MAE 6.50, automatically, without a new hyperparameter. Weights
are computed from observed performance, not assumptions.

---

## Approach

### 1. Data Exploration

5.2M rows, 481 trading days, 200 stocks.

Key findings:
- Target centered near zero (mean: -0.05 bps) with fat tails ranging ±400 bps
- Middle period (200–400s) has highest volatility, MAE 22% higher than late period
- Smaller stocks (low index weight) are substantially harder to predict
- Buy/sell imbalance magnitude drops sharply toward auction close

### 2. Stock Weight Estimation

Estimated index weights from `matched_size` as a proxy for market cap rather than
using hardcoded values:

```
Top 10 lead stocks: [112, 45, 168, 191, 41, 175, 179, 84, 142, 95]
Weight range: 0.007% to 9.96%
```

### 3. Stock Clustering

KMeans clustering (9 clusters) by behavioral profile: WAP volatility, target std,
estimated weight, spread.

Notable clusters:
- **Clusters 2 & 4:** High-weight stocks, lower volatility, contains all 8 lead stocks
- **Cluster 5:** 3 tiny stocks with extreme volatility (target std: 20)
- **Cluster 2:** Just 2 mega-cap stocks comprising ~19% of index weight

### 4. Feature Engineering (115 features → 107 after selection)

| Category | Features |
|---|---|
| Basic | spread, volume, liquidity imbalance, price comparisons |
| Time-based | lag features (1,2,3,5,10 periods), returns, diffs |
| Cross-sectional | performance vs market WAP, vs weighted index, percentile ranks |
| Domain-specific | inferred price from tick size, revealed targets, weighted index returns |

Feature selection via gain-based importance on a proxy LGB model reduced 115
features to 107, cutting training time ~35% with negligible MAE impact.

### 5. Model Architecture

Four-model ensemble trained with purged time-series CV (4 folds, 5-day purge gap):

| Model | Purpose |
|---|---|
| XGBoost | MAE objective, selected features |
| LightGBM GBDT | MAE objective, early stopping |
| LightGBM DART | Dropout regularization, improved tail prediction |
| Quantile Blend | Pinball loss at q=[0.1, 0.25, 0.5, 0.75, 0.9], adaptive weighting |

All combined via inverse-MAE weighting, then zero-sum adjusted per timestamp.

---

## Optimization: Addressing Tail Collapse

Standard MAE-optimized models predicted moves of ±6 bps while actuals reached
±28 bps. Two methods addressed this:

**DART (Dropout meets Additive Regression Trees)**
Standard GBDT over-relies on dominant early trees, pulling predictions toward the
mean. DART randomly drops a fraction of existing trees during training (drop_rate=0.1),
forcing independent tree contributions. Extended prediction range from ±6.06 to ±6.09
bps at p99, modest but directionally correct.

**Quantile Regression with Adaptive Blending**
Five LightGBM models trained at q=[0.1, 0.25, 0.5, 0.75, 0.9] using pinball loss.
The q=0.9 model is penalized 9× harder for under-predicting than over-predicting,
structurally forcing it to learn the upper tail.

Blending weights adapt dynamically using `signed_imbalance` from the order book:
when buy pressure is high, weights shift toward q=0.75 and q=0.9; sell pressure
shifts toward q=0.1 and q=0.25. Extended prediction range to ±7.32 bps at p99.

---

## Results (4-fold purged time-series CV, full dataset 5.2M rows)

### Model Comparison

| Model | Mean MAE | Std | Best Fold | Worst Fold |
|---|---|---|---|---|
| XGBoost | 6.429 | 0.398 | 5.928 | 7.041 |
| LightGBM GBDT | 6.430 | 0.399 | 5.930 | 7.045 |
| LightGBM DART | 6.433 | 0.401 | 5.937 | 7.046 |
| 4-way Ensemble | 6.432 | 0.396 | 5.936 | 7.042 |
| **Adjusted (final)** | **6.428** | **0.393** | **5.938** | **7.034** |

### Period Breakdown

| Period | MAE | Notes |
|---|---|---|
| Early (0–200s) | 6.300 | Book noisy but bounded |
| **Middle (200–400s)** | **7.246** | Highest volatility, hardest to predict |
| Late (400–600s) | 5.510 | Book stabilizes, intentions revealed |

### Tail Coverage

| Percentile | Actual | Standard LGB | DART | Quantile Blend |
|---|---|---|---|---|
| p1 | -27.98 bps | -5.70 | -5.59 | -7.27 |
| p5 | -15.82 bps | -3.22 | -3.06 | -4.49 |
| p95 | +15.71 bps | +3.08 | +3.09 | +4.09 |
| p99 | +28.44 bps | +6.06 | +6.09 | +7.32 |

**Interval coverage [q10, q90]: 79–82% across folds** (theoretical target: 80%)

### What Worked

- **Lead stocks are substantially easier:** MAE ~4.3 vs ~5.9 for other stocks
- **`seconds_in_bucket` is the top feature** in LightGBM, time within the auction
  window dominates all other signals
- **Zero-sum adjustment:** free improvement at inference time, no training cost
- **Feature selection:** 107 vs 115 features with no meaningful MAE change

### What's Still Hard

- **Middle period (200–400s):** MAE 22% higher than late period, rapid book
  cancellations make this window structurally hard to model
- **Volatile small-caps:** Clusters 5 and 8 have MAE of 15–20
- **Tail collapse persists:** quantile blend reaches ±7 bps against actual ±28 bps,
  meaningful improvement but the gap remains large

### Top Features

**XGBoost:** `size_imbalance`, `liquidity_imbalance`, `reference_price_wap_imb`

**LightGBM:** `seconds_in_bucket`, `market_imbalance`, `market_flag`, `stock_wap_volatility`

---

## What I'd Try Next

1. Sequence model (LSTM or Transformer) treating the 600-second window as a time
   series, the current model treats each row independently
2. Stock-specific models for volatile clusters (5 and 8) where global models fail
3. Stacking meta-learner instead of inverse-MAE weighting to better balance the
   four ensemble members
4. More aggressive feature selection to reduce correlated features and further
   cut training time

---

## Acknowledgments

Ideas borrowed from:
- [lognorm's winning solution](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion),  index weights, inferred price, revealed targets
- Various public notebooks for triplet imbalance and baseline features

---

*Competition: [Optiver - Trading at the Close (Kaggle 2023-2024)](https://www.kaggle.com/competitions/optiver-trading-at-the-close)*
