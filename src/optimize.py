
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import json
from pathlib import Path


# ── Feature selection ────────────────────────────────────────────────────────

FAST_LGB_PARAMS = {
    'learning_rate': 0.05,       # faster than production rate
    'max_depth': 6,
    'n_estimators': 300,
    'num_leaves': 64,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'mae',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    importance_threshold: float = 0.0001,
    save_path: str = 'results/selected_features.json',
) -> list[str]:
    """
    Train a fast proxy LGB model and return features above an importance threshold.

    Uses gain-based importance (not split count) because gain measures how much
    each feature actually reduces the loss, not just how often it is used.

    Parameters
    ----------
    importance_threshold : Features with mean importance below this fraction of
                           the top feature's importance are dropped.
                           0.0001 = drop anything contributing less than 0.01% of
                           the top feature's contribution. Conservative by design.

    Returns
    -------
    List of selected feature names, ordered by importance descending.
    """
    print("Running feature selection...")

    proxy = lgb.LGBMRegressor(**FAST_LGB_PARAMS)
    proxy.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    importances = pd.Series(
        proxy.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    # Normalise so top feature = 1.0
    importances_norm = importances / (importances.max() + 1e-9)

    selected = importances_norm[importances_norm >= importance_threshold].index.tolist()
    dropped = importances_norm[importances_norm < importance_threshold].index.tolist()

    proxy_val_preds = proxy.predict(X_val)
    proxy_mae = mean_absolute_error(y_val, proxy_val_preds)

    print(f"  Features: {len(X_train.columns)} → {len(selected)} selected, {len(dropped)} dropped")
    print(f"  Proxy model MAE (all features): {proxy_mae:.5f}")
    print(f"  Top 10 features:")
    for feat, imp in importances_norm.head(10).items():
        print(f"    {feat:<45} {imp:.4f}")

    # Persist for reproducibility and README documentation
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'selected': selected,
            'dropped': dropped,
            'n_selected': len(selected),
            'n_dropped': len(dropped),
            'proxy_mae': float(proxy_mae),
            'top_features': importances_norm.head(20).to_dict(),
        }, f, indent=2)

    return selected


# ── DART configuration ───────────────────────────────────────────────────────

DART_LGB_PARAMS = {
    'boosting_type': 'dart',     # key change: dropout regularization
    'learning_rate': 0.05,       # DART needs higher LR — no early stopping possible
    'max_depth': 8,
    'n_estimators': 1000,        # fixed; DART cannot use early stopping
    'num_leaves': 128,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'drop_rate': 0.1,            # fraction of trees dropped per round
    'skip_drop': 0.5,            # probability of skipping dropout entirely
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'mae',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def train_dart_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple:
    """
    Train a LightGBM model in DART mode.

    Why DART improves on extreme values:
    Standard gradient boosting adds trees greedily — later trees correct errors
    of earlier ones, which causes the model to over-rely on a small subset of
    high-influence trees. These dominant trees pull predictions toward the mean.

    DART randomly drops a fraction of existing trees during each boosting round,
    forcing new trees to be useful independently rather than as corrections.
    This distributes predictive contribution more evenly and reduces the
    shrinkage effect that collapses predictions toward zero.

    Limitation: DART cannot use early stopping (the dropout makes validation
    loss non-monotonic). n_estimators is therefore fixed.
    """
    print("  Training LightGBM DART...")

    model = lgb.LGBMRegressor(**DART_LGB_PARAMS)
    # Note: no early_stopping callback — incompatible with DART
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"  DART MAE: {mae:.5f}")

    return model, preds, mae


# ── Comparison utility ───────────────────────────────────────────────────────

def compare_prediction_ranges(
    y_true: np.ndarray,
    standard_preds: np.ndarray,
    dart_preds: np.ndarray,
) -> dict:
    """
    Compare how well each model covers the extreme tails of the target.

    This is the key metric for evaluating DART's benefit — not just MAE
    but whether predictions actually reach the tails of the distribution.
    """
    percentiles = [1, 5, 25, 50, 75, 95, 99]

    results = {
        'actual': {f'p{p}': float(np.percentile(y_true, p)) for p in percentiles},
        'standard': {f'p{p}': float(np.percentile(standard_preds, p)) for p in percentiles},
        'dart': {f'p{p}': float(np.percentile(dart_preds, p)) for p in percentiles},
    }

    print("\n  Prediction range comparison:")
    print(f"  {'Percentile':<12} {'Actual':>10} {'Standard':>10} {'DART':>10}")
    print("  " + "-" * 45)
    for p in percentiles:
        key = f'p{p}'
        print(f"  p{p:<11} {results['actual'][key]:>10.2f} "
              f"{results['standard'][key]:>10.2f} "
              f"{results['dart'][key]:>10.2f}")

    return results


# ── Quantile Regression ──────────────────────────────────────────────────────

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# Base params for quantile models — lighter than production to keep training fast
QUANTILE_LGB_BASE = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': 7,
    'n_estimators': 500,
    'num_leaves': 64,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Pinball (quantile) loss for a single quantile.
    Equivalent to MAE but asymmetric: penalises over-prediction more at low quantiles
    and under-prediction more at high quantiles.
    """
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)))


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: list = None,
) -> tuple[dict, dict, np.ndarray]:
    """
    Train one LightGBM model per quantile using pinball loss.

    Why multiple quantiles instead of just predicting the mean:
    A mean-optimised model minimises symmetric squared or absolute error,
    which pulls predictions toward the centre of the distribution. Pinball
    loss is asymmetric — the q=0.9 model is penalised 9x more for
    under-predicting than over-predicting, forcing it to learn the upper tail.
    Training across multiple quantiles gives us a full conditional distribution
    estimate, not just the conditional mean.

    Returns
    -------
    models       : {quantile: fitted LGBMRegressor}
    pinball_scores : {quantile: pinball loss on validation set}
    median_preds : predictions from the q=0.5 model (comparable to MAE model)
    """
    if quantiles is None:
        quantiles = QUANTILES

    print(f"  Training {len(quantiles)} quantile models...")
    models = {}
    pinball_scores = {}
    median_preds = None

    for q in quantiles:
        params = {
            **QUANTILE_LGB_BASE,
            'objective': 'quantile',
            'alpha': q,          # LightGBM's parameter name for the quantile
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        score = _pinball_loss(y_val.values, preds, q)
        models[q] = model
        pinball_scores[q] = score

        if abs(q - 0.5) < 1e-6:
            median_preds = preds

        print(f"    q={q:.2f}  pinball={score:.5f}  "
              f"pred_range=[{preds.min():.2f}, {preds.max():.2f}]")

    return models, pinball_scores, median_preds


def adaptive_quantile_blend(
    X_val: pd.DataFrame,
    quantile_models: dict,
    imbalance_feature: str = 'signed_imbalance',
) -> np.ndarray:
    """
    Blend quantile predictions using adaptive weights based on order book imbalance.

    Core idea: when the order book shows a large signed imbalance (strong buy or
    sell pressure), extreme price moves become more likely. In those cases we
    should weight the tail quantiles (q=0.1 or q=0.9) more heavily. When the
    book is balanced, the median (q=0.5) is most reliable.

    Weighting scheme:
    - Compute a normalised imbalance signal in [-1, 1]
    - Positive signal (buy pressure) → up-weight q=0.75 and q=0.9
    - Negative signal (sell pressure) → up-weight q=0.1 and q=0.25
    - Near-zero signal → concentrate weight on q=0.5

    This is a simple but principled way to use the distribution information
    without requiring a separate meta-learner.

    Parameters
    ----------
    imbalance_feature : column in X_val to use as the imbalance signal.
                        Falls back to equal weighting if column not found.
    """
    quantiles = sorted(quantile_models.keys())
    preds_matrix = np.column_stack([
        quantile_models[q].predict(X_val) for q in quantiles
    ])   # shape: (n_samples, n_quantiles)

    if imbalance_feature not in X_val.columns:
        # fallback: simple average across quantiles
        print("    Imbalance feature not found — using equal quantile weights")
        return preds_matrix.mean(axis=1)

    # Normalise imbalance to [-1, 1] using tanh (soft clipping)
    raw_imbalance = X_val[imbalance_feature].values
    scale = np.percentile(np.abs(raw_imbalance), 95) + 1e-6
    signal = np.tanh(raw_imbalance / scale)   # shape: (n_samples,)

    # Build weight matrix: shape (n_samples, n_quantiles)
    # Base weight: uniform across quantiles
    n_q = len(quantiles)
    weights = np.ones((len(signal), n_q)) / n_q

    q_arr = np.array(quantiles)   # e.g. [0.1, 0.25, 0.5, 0.75, 0.9]

    for i, s in enumerate(signal):
        if s > 0:
            # Buy pressure: shift weight toward upper quantiles
            # Weight proportional to quantile value, scaled by signal strength
            upper_bonus = q_arr * abs(s)
            w = 1/n_q + upper_bonus
        else:
            # Sell pressure: shift weight toward lower quantiles
            lower_bonus = (1 - q_arr) * abs(s)
            w = 1/n_q + lower_bonus
        weights[i] = w / w.sum()   # normalise to sum to 1

    blended = (preds_matrix * weights).sum(axis=1)
    return blended


def evaluate_quantile_coverage(
    y_true: np.ndarray,
    quantile_models: dict,
    X_val: pd.DataFrame,
    standard_preds: np.ndarray,
    dart_preds: np.ndarray,
    blended_preds: np.ndarray,
) -> dict:
    """
    Compare tail coverage across all prediction methods.

    Reports:
    - Prediction range at key percentiles for all methods vs actuals
    - MAE of the blended quantile prediction vs standard and DART
    - Interval coverage: what % of actuals fall within [q10, q90] predictions
    """
    percentiles = [1, 5, 25, 50, 75, 95, 99]

    print("\n  Full prediction range comparison:")
    print(f"  {'Pct':<6} {'Actual':>8} {'Standard':>10} {'DART':>8} {'Quantile':>10}")
    print("  " + "-" * 46)

    results = {}
    for p in percentiles:
        a  = float(np.percentile(y_true, p))
        s  = float(np.percentile(standard_preds, p))
        d  = float(np.percentile(dart_preds, p))
        q  = float(np.percentile(blended_preds, p))
        results[f'p{p}'] = {'actual': a, 'standard': s, 'dart': d, 'quantile_blend': q}
        print(f"  p{p:<5} {a:>8.2f} {s:>10.2f} {d:>8.2f} {q:>10.2f}")

    # MAE comparison
    mae_standard = mean_absolute_error(y_true, standard_preds)
    mae_dart     = mean_absolute_error(y_true, dart_preds)
    mae_blend    = mean_absolute_error(y_true, blended_preds)
    print(f"\n  MAE — Standard: {mae_standard:.5f} | DART: {mae_dart:.5f} | Quantile blend: {mae_blend:.5f}")

    # Interval coverage: % of actuals inside [q10_pred, q90_pred]
    if 0.1 in quantile_models and 0.9 in quantile_models:
        lower = quantile_models[0.1].predict(X_val)
        upper = quantile_models[0.9].predict(X_val)
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        print(f"  Interval coverage [q10, q90]: {coverage:.1%}  (ideal ≈ 80%)")
        results['interval_coverage_q10_q90'] = coverage

    results['mae_standard'] = mae_standard
    results['mae_dart']     = mae_dart
    results['mae_blend']    = mae_blend

    return results
