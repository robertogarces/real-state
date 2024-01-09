import numpy as np
import pandas as pd

def mape_score(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true (list or array): Actual values.
    - y_pred (list or array): Predicted values.

    Returns:
    - float: Mean Absolute Percentage Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")

    mape = sum(abs((true - pred) / true) for true, pred in zip(y_true, y_pred)) / len(y_true)

    return mape


def model_evaluation(test_set, model, target, decimals=2, log_inverse_transform=False):

    test_preds = model.predict(test_set.drop(columns=target))
    test_actual = test_set[target]

    if log_inverse_transform==True:
        test_preds = np.exp(test_preds)
        test_actual = np.exp(test_actual)

    rmse = np.sqrt(np.mean((test_actual - test_preds)**2))
    mse = np.mean((test_actual - test_preds)**2)
    mae = np.mean(np.abs(test_actual - test_preds))
    median_absolute_error = np.median(np.abs(test_actual - test_preds))
    mape = mape_score(test_actual, test_preds)
    r2 = np.corrcoef(test_actual, test_preds)[0, 1]**2

    print(f'RMSE: {round(rmse, decimals)}')
    print(f'MSE : {round(mse, decimals)}')
    print(f'MAE : {round(mae, decimals)}')
    print(f'MeAE: {round(median_absolute_error, decimals)}')
    print(f'MAPE: {round(mape, decimals)}')
    print(f'R2  : {round(r2, decimals)}')

    return r2, mape, rmse, median_absolute_error