# src/evaluation.py
import numpy as np
import json
from pathlib import Path


def print_final_summary(fold_results: list) -> None:
    """Print a formatted cross-validation results table to stdout."""
    print("=" * 70)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("=" * 70)

    xgb_maes = [r['xgb_mae'] for r in fold_results]
    lgb_maes = [r['lgb_mae'] for r in fold_results]
    ens_maes = [r['ensemble_mae'] for r in fold_results]
    adj_maes = [r['adjusted_mae'] for r in fold_results]

    print(f"\n{'Model':<15} {'Mean MAE':<12} {'Std':<10} {'Best':<10} {'Worst':<10}")
    print("-" * 60)
    for name, maes in [('XGBoost', xgb_maes), ('LightGBM', lgb_maes),
                        ('Ensemble', ens_maes), ('Adjusted', adj_maes)]:
        print(f"{name:<15} {np.mean(maes):<12.5f} {np.std(maes):<10.5f} "
              f"{min(maes):<10.5f} {max(maes):<10.5f}")

    print("\n" + "-" * 60)
    print("PERIOD BREAKDOWN (Adjusted)")
    print("-" * 60)
    for period in ['early', 'middle', 'late']:
        period_maes = [r[f'{period}_mae'] for r in fold_results]
        print(f"{period.capitalize():<10} {np.mean(period_maes):.5f} +/- {np.std(period_maes):.5f}")

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)


def _make_serialisable(v):
    """Recursively convert a value to a JSON-safe Python type. Drops anything unrecognised."""
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {kk: _make_serialisable(vv) for kk, vv in v.items()}
    if isinstance(v, list):
        return [_make_serialisable(i) for i in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return None  # silently drop models, DataFrames, or anything else non-serialisable


def save_results(fold_results: list, output_path: str = "results/cv_results.json") -> None:
    """
    Persist CV metrics to JSON so results are reproducible without re-running training.
    Model objects, arrays, and DataFrames are excluded automatically.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    skip_keys = {
        'xgb_model', 'lgb_model', 'dart_model', 'quantile_models',
        'xgb_preds', 'lgb_preds', 'dart_preds', 'quantile_blend',
        'ensemble_preds', 'adjusted_preds',
        'val_df', 'y_val',
    }

    serialisable = []
    for i, r in enumerate(fold_results):
        fold_dict = {'fold': i + 1}
        for k, v in r.items():
            if k in skip_keys:
                continue
            fold_dict[k] = _make_serialisable(v)
        serialisable.append(fold_dict)

    with open(output_path, 'w') as f:
        json.dump(serialisable, f, indent=2)

    print(f"Results saved to {output_path}")
