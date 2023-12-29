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
