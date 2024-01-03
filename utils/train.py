import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd
from .evaluation import mape_score


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd
from .evaluation import mape_score

def train_lightgbm_model(dataframe, target, best_params=None):

    X = dataframe.drop(target, axis=1)
    y = dataframe[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if best_params is None:
        # Parámetros iniciales del modelo
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
            # Agrega otros parámetros que desees ajustar aquí
        }

        # Entrenamiento del modelo
        model = lgb.train(params,
                          lgb.Dataset(X_train, label=y_train),
                          num_boost_round=1000,
                          valid_sets=[lgb.Dataset(X_val, label=y_val)],
                       #   early_stopping_rounds=50
                       )
    else:
        # Modelo con los mejores parámetros proporcionados
        params = best_params
        model = lgb.train(params,
                          lgb.Dataset(X_train, label=y_train),
                          num_boost_round=1000,
                          valid_sets=[lgb.Dataset(X_val, label=y_val)],
                       #   early_stopping_rounds=50
                       )

    # Evaluación en el conjunto de validación
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)

    test_rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    test_mse = np.mean((y_val - y_pred)**2)
    test_mae = np.mean(np.abs(y_val - y_pred))
    test_median_absolute_error = np.median(np.abs(y_val - y_pred))
    test_r2 = np.corrcoef(y_val, y_pred)[0, 1]**2    
    mape = mape_score(y_val, y_pred)

    print(f'RMSE: {int(test_rmse)}')
    print(f'MSE : {int(test_mse)}')
    print(f'MAE : {int(test_mae)}')
    print(f'MeAE: {int(test_median_absolute_error)}')
    print(f'MAPE: {int(mape)}')
    print(f'R2  : {round(test_r2, 3)}')

    return model



def train_lightgbm_model_(dataframe, target, best_params=None):

    X = dataframe.drop(target, axis=1)
    y = dataframe[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if best_params==None:
        model = lgb.LGBMRegressor(n_estimators=1000)
    else:
        model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)

    # Evaluación en el conjunto de validación
    y_pred = model.predict(X_val)

    test_rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    test_mse = np.mean((y_val - y_pred)**2)
    test_mae = np.mean(np.abs(y_val - y_pred))
    test_median_absolute_error = np.median(np.abs(y_val - y_pred))
    test_r2 = np.corrcoef(y_val, y_pred)[0, 1]**2    
    mape = mape_score(y_val, y_pred)

    print(f'RMSE: {int(test_rmse)}')
    print(f'MSE : {int(test_mse)}')
    print(f'MAE : {int(test_mae)}')
    print(f'MeAE: {int(test_median_absolute_error)}')
    print(f'MAPE: {int(mape)}')
    print(f'R2  : {round(test_r2, 3)}')

    return model

