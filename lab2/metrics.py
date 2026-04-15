"""
Пользовательские функции вычисления метрик регрессии (numpy).
Подключается к основному ноутбуку: import metrics as cm
"""
import numpy as np


def r2(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def all_metrics(y_true, y_pred):
    return {
        "R2":   round(r2(y_true, y_pred), 4),
        "MSE":  round(mse(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE":  round(mae(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
    }
