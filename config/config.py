# Config file

TARGET = 'price'   # or price_log
if TARGET == 'price_log':
    TARGET = 1
else:
    TARGET = 0

SEED = 42

LOWER_BOUND = 1.5
UPPER_BOUND = 98.5

TRAIN_SIZE = 0.7
INFERENCE_SIZE = 0.05
TEST_SIZE = 1 - (TRAIN_SIZE - INFERENCE_SIZE)
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
LOG_INVERSE_TRANSFORM = False


