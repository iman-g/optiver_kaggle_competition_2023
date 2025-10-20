# Optiver Kaggle Competition 2023 🧠📈
My solution for the [Optiver Trading at the Close (2023)](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview) Kaggle competition.  
The competition focuses on predicting the **target value** for financial time series data at the close of trading, based on order book and trade features.

---

## 📂 Competition Overview
The dataset and competition details are available here:  
🔗 [Competition Page](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)  
🔗 [Dataset](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data)

The goal is to build a model that predicts **future price movements** using limit order book data for multiple stocks across different time intervals.

---

## 🧰 Project Structure

---

### ⚙️ Data Preprocessing Pipeline

1. **Feature Engineering**
   - Created rolling statistics of WAP (weighted average price), spread, and volatility at stock level.
   - Derived imbalance ratios, matched sizes, and directional indicators.
   - Aggregated stock-level statistics such as mean, standard deviation, and skewness.

2. **Missing Value Handling**
   - Numerical columns filled using **median imputation**.
   - Group-level imputation based on `stock_id` where appropriate.

3. **Normalization**
   - Standard scaling applied to features with different magnitudes for better model convergence.

4. **Train–Validation Split**
   - The training data was split **90% train / 10% validation** to evaluate performance locally.

---

### 🤖 Models Used

| Model | Library | Key Hyperparameters |
|--------|----------|--------------------|
| LightGBM | `lightgbm.LGBMRegressor` | `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31` |
| XGBoost | `xgboost.XGBRegressor` | `n_estimators=500`, `learning_rate=0.05`, `max_depth=6` |
| HistGradientBoosting | `sklearn.ensemble.HistGradientBoostingRegressor` | `max_iter=500`, `learning_rate=0.05` |

Each base model was trained on the processed features and evaluated using **Mean Absolute Error (MAE)**.

---

### 🧩 Ensemble Model (Stacking)

The predictions from the three base models were combined using a **meta-model (Linear Regression)** trained on out-of-fold predictions.  
This stacking approach improved the stability and accuracy of the final predictions.

---

### 📊 Evaluation

- Metric: **Mean Absolute Error (MAE)**
- Validation Strategy: Time-aware split based on `date_id` and `time_id`
- Observed improvement: Ensemble model reduced MAE by ~3–5% compared to individual models.

---



