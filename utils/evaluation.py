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


def evaluate_test_set(test_set, model, target, decimals=0, log_inverse_transform=False):

    test_preds = model.predict(test_set.drop(columns=target))
    test_actual = test_set[target]

    if log_inverse_transform==True:
        test_preds = np.exp(test_preds)
        test_actual = np.exp(test_actual)

    test_rmse = np.sqrt(np.mean((test_actual - test_preds)**2))
    test_mse = np.mean((test_actual - test_preds)**2)
    test_mae = np.mean(np.abs(test_actual - test_preds))
    test_median_absolute_error = np.median(np.abs(test_actual - test_preds))
    test_mape = mape_score(test_actual, test_preds)
    test_r2 = np.corrcoef(test_actual, test_preds)[0, 1]**2

    print(f'RMSE: {round(test_rmse, decimals)}')
    print(f'MSE : {round(test_mse, decimals)}')
    print(f'MAE : {round(test_mae, decimals)}')
    print(f'MeAE: {round(test_median_absolute_error, decimals)}')
    print(f'MAPE: {round(test_mape, decimals)}')
    print(f'R2  : {round(test_r2, decimals)}')

