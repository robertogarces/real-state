# Config file

TARGET = 'price'   # or price_log
if TARGET == 'price_log':
    TARGET = 1
else:
    TARGET = 0

SEED = 42
TEST_SIZE = 0.15
USE_OPTUNA = True
N_TRIALS = 3
SAVE_OPTIMIZED_PARAMS = True
USE_OPTIMIZED_PARAMS = True

DEFAULT_LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'n_estimators': 1000
    }

NUM_BOOST_ROUND = 1000
EVALUATION_METRIC_DECIMALS = 3
LOG_INVERSE_TRANSFORM = True


