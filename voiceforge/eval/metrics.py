"""度量函数：分类(accuracy/f1) 与 回归(rmse/r2)。

注意：sklearn 同名函数导入为 _sk_* 别名，避免评测函数内递归调用自身。
"""
import numpy as np
from sklearn.metrics import accuracy_score as _sk_accuracy
from sklearn.metrics import f1_score as _sk_f1
from sklearn.metrics import mean_squared_error as _sk_mse
from sklearn.metrics import r2_score as _sk_r2


def accuracy(y_true, y_pred) -> float:
    return float(_sk_accuracy(list(y_true), list(y_pred)))


def f1_macro(y_true, y_pred) -> float:
    return float(_sk_f1(list(y_true), list(y_pred), average="macro", zero_division=0))


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(_sk_mse(yt, yp)))


def r2(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(_sk_r2(yt, yp))
