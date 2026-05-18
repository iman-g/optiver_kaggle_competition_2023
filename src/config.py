
# Color palette for visualizations
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#3B3B3B',
}

# Training configuration
CONFIG = {
    'n_splits': 5,
    'purge_days': 5,
    'xgb_estimators': 5000,
    'lgb_estimators': 5000,
    'early_stopping': 200,
}

# Model hyperparameters
XGB_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 10,
    'n_estimators': CONFIG['xgb_estimators'],
    'min_child_weight': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:absoluteerror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}

LGB_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 10,
    'n_estimators': CONFIG['lgb_estimators'],
    'num_leaves': 256,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'mae',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}
