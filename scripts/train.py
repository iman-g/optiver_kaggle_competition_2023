# scripts/train.py
"""
Full training pipeline for Optiver Trading at the Close.

Usage:
    python -m scripts.train --data train.csv
    python -m scripts.train --data train.csv --rapid        # fast smoke-test
    python -m scripts.train --data train.csv --no-optimize  # baseline only
"""
import argparse
import gc
import time

import numpy as np
from sklearn.metrics import mean_absolute_error

from src.config import CONFIG, XGB_PARAMS, LGB_PARAMS
from src.data import load_data, estimate_stock_weights, create_stock_profiles, cluster_stocks
from src.features import create_features, create_revealed_targets
from src.model import (PurgedGroupTimeSeriesSplit, prepare_train_val,
                       train_fold, weighted_zero_sum_adjustment)
from src.optimize import (select_features, train_dart_model,
                          train_quantile_models, adaptive_quantile_blend,
                          evaluate_quantile_coverage)
from src.evaluation import print_final_summary, save_results
from src.visualization import plot_model_comparison, plot_detailed_fold_analysis

import xgboost as xgb
import lightgbm as lgb


def train_fold_optimized(
    X_train, y_train, X_val, y_val,
    val_df, stock_weights: dict,
    fold: int,
    selected_features: list,
) -> dict:
    """
    Four-model ensemble:
      1. XGBoost          (MAE objective, selected features)
      2. LightGBM GBDT    (MAE objective, early stopping)
      3. LightGBM DART    (dropout regularization, fixed trees)
      4. Quantile blend   (adaptive weighting across q10/q25/q50/q75/q90)

    All combined via inverse-MAE weighting, then zero-sum adjusted.
    """
    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]

    # ── 1. XGBoost ───────────────────────────────────────────────────
    print(f"\n  Training XGBoost ({len(selected_features)} features)...")
    xgb_model = xgb.XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=CONFIG['early_stopping'],
    )
    xgb_model.fit(
        X_train_sel, y_train,
        eval_set=[(X_val_sel, y_val)],
        verbose=100,
    )
    xgb_preds = xgb_model.predict(X_val_sel)
    xgb_mae   = mean_absolute_error(y_val, xgb_preds)
    print(f"  XGBoost MAE: {xgb_mae:.5f}")

    # ── 2. Standard LightGBM ─────────────────────────────────────────
    print(f"\n  Training LightGBM ({len(selected_features)} features)...")
    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_model.fit(
        X_train_sel, y_train,
        eval_set=[(X_val_sel, y_val)],
        callbacks=[lgb.early_stopping(CONFIG['early_stopping']), lgb.log_evaluation(100)],
    )
    lgb_preds = lgb_model.predict(X_val_sel)
    lgb_mae   = mean_absolute_error(y_val, lgb_preds)
    print(f"  LightGBM MAE: {lgb_mae:.5f}")

    # ── 3. DART LightGBM ─────────────────────────────────────────────
    dart_model, dart_preds, dart_mae = train_dart_model(
        X_train_sel, y_train, X_val_sel, y_val
    )

    # ── 4. Quantile regression blend ─────────────────────────────────
    print(f"\n  Training quantile models...")
    quantile_models, pinball_scores, median_preds = train_quantile_models(
        X_train_sel, y_train, X_val_sel, y_val
    )
    quantile_blend = adaptive_quantile_blend(X_val_sel, quantile_models)
    quantile_mae   = mean_absolute_error(y_val, quantile_blend)
    print(f"  Quantile blend MAE: {quantile_mae:.5f}")

    # ── Coverage evaluation (the key diagnostic) ─────────────────────
    coverage_results = evaluate_quantile_coverage(
        y_val.values, quantile_models, X_val_sel,
        lgb_preds, dart_preds, quantile_blend,
    )

    # ── Four-way inverse-MAE ensemble ────────────────────────────────
    w_xgb      = 1 / (xgb_mae      + 1e-6)
    w_lgb      = 1 / (lgb_mae      + 1e-6)
    w_dart     = 1 / (dart_mae     + 1e-6)
    w_quantile = 1 / (quantile_mae + 1e-6)
    w_total    = w_xgb + w_lgb + w_dart + w_quantile

    ensemble_preds = (
        w_xgb      * xgb_preds     +
        w_lgb      * lgb_preds     +
        w_dart     * dart_preds    +
        w_quantile * quantile_blend
    ) / w_total
    ensemble_mae = mean_absolute_error(y_val, ensemble_preds)

    print(f"\n  4-way Ensemble MAE: {ensemble_mae:.5f}")
    print(f"  Weights — XGB: {w_xgb/w_total:.3f}  LGB: {w_lgb/w_total:.3f}  "
          f"DART: {w_dart/w_total:.3f}  Quantile: {w_quantile/w_total:.3f}")

    # ── Zero-sum post-processing ──────────────────────────────────────
    adjusted_preds = weighted_zero_sum_adjustment(ensemble_preds, val_df, stock_weights)
    adjusted_mae   = mean_absolute_error(y_val, adjusted_preds)
    print(f"  Adjusted MAE: {adjusted_mae:.5f}")

    return {
        # models
        'xgb_model':        xgb_model,
        'lgb_model':        lgb_model,
        'dart_model':       dart_model,
        'quantile_models':  quantile_models,
        # predictions
        'xgb_preds':        xgb_preds,
        'lgb_preds':        lgb_preds,
        'dart_preds':       dart_preds,
        'quantile_blend':   quantile_blend,
        'ensemble_preds':   ensemble_preds,
        'adjusted_preds':   adjusted_preds,
        # scores
        'xgb_mae':          xgb_mae,
        'lgb_mae':          lgb_mae,
        'dart_mae':         dart_mae,
        'quantile_mae':     quantile_mae,
        'ensemble_mae':     ensemble_mae,
        'adjusted_mae':     adjusted_mae,
        # weights
        'xgb_weight':       w_xgb      / w_total,
        'lgb_weight':       w_lgb      / w_total,
        'dart_weight':      w_dart     / w_total,
        'quantile_weight':  w_quantile / w_total,
        # diagnostics
        'coverage_results': coverage_results,
        'pinball_scores':   pinball_scores,
        'n_features':       len(selected_features),
    }


def main(data_path: str, rapid: bool = False, optimize: bool = True):

    t_start = time.time()

    # 1. Load
    df = load_data(data_path, rapid_mode=rapid)

    # 2. Stock metadata
    stock_weights, lead_stocks, _ = estimate_stock_weights(df)
    stock_profiles = create_stock_profiles(df, stock_weights, lead_stocks)
    stock_profiles, cluster_summary, _ = cluster_stocks(stock_profiles, n_clusters=9)
    print(cluster_summary)

    # 3. Features
    revealed_targets = create_revealed_targets(df)
    df = create_features(df, stock_profiles, stock_weights, lead_stocks, revealed_targets)
    print(f"Final dataset shape: {df.shape}")

    # 4. Cross-validation
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=CONFIG['n_splits'],
        purge_gap=CONFIG['purge_days'],
    )
    date_ids         = df['date_id']
    fold_results     = []
    selected_features = None   # selected on fold 1, reused thereafter

    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, date_ids), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{CONFIG['n_splits']}")
        print(f"{'='*60}")

        X_train, y_train, X_val, y_val, val_df_fold, feature_cols = prepare_train_val(
            df, train_idx, val_idx, stock_profiles
        )
        print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Features: {len(feature_cols)}")

        # Feature selection: once on fold 1, applied to all folds
        if optimize and selected_features is None:
            selected_features = select_features(
                X_train, y_train, X_val, y_val,
                importance_threshold=0.0001,
            )

        if optimize:
            results = train_fold_optimized(
                X_train, y_train, X_val, y_val,
                val_df_fold, stock_weights, fold, selected_features,
            )
        else:
            results = train_fold(
                X_train, y_train, X_val, y_val,
                val_df_fold, stock_weights, fold,
            )

        # Period-wise MAE
        for period, (t_min, t_max) in [
            ('early',  (0,   200)),
            ('middle', (200, 400)),
            ('late',   (400, 600)),
        ]:
            mask = ((val_df_fold['seconds_in_bucket'] >= t_min) &
                    (val_df_fold['seconds_in_bucket'] <  t_max))
            results[f'{period}_mae'] = mean_absolute_error(
                y_val[mask], results['adjusted_preds'][mask]
            )

        results['val_df'] = val_df_fold
        results['y_val']  = y_val
        fold_results.append(results)
        gc.collect()

    # 5. Results
    print_final_summary(fold_results)
    save_results(fold_results)
    print(f"\nTotal runtime: {(time.time() - t_start)/60:.1f} minutes")

    # 6. Visualisation
    plot_model_comparison(fold_results)
    plot_detailed_fold_analysis(fold_results, stock_profiles, lead_stocks, fold_idx=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optiver closing price prediction pipeline")
    parser.add_argument('--data',        required=True,       help="Path to train.csv")
    parser.add_argument('--rapid',       action='store_true', help="Use 5%% of dates for fast iteration")
    parser.add_argument('--no-optimize', action='store_true', help="Baseline only, no DART/quantile")
    args = parser.parse_args()

    main(
        data_path=args.data,
        rapid=args.rapid,
        optimize=not args.no_optimize,
    )
