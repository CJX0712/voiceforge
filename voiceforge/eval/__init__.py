"""eval: 标准度量。sklearn 导入改名以避免与本地同名函数递归。"""
from .metrics import accuracy, f1_macro, rmse, r2

__all__ = ["accuracy", "f1_macro", "rmse", "r2"]
