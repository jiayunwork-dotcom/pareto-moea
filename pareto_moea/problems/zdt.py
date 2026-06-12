"""ZDT系列测试函数"""

import numpy as np
from .base import Problem


class ZDTBase(Problem):
    """ZDT系列基类"""

    def __init__(self, n_var: int = 30, name: str = "ZDT"):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, 2, xl, xu, name)


class ZDT1(ZDTBase):
    """ZDT1 - 凸帕累托前沿"""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var, "ZDT1")
        self.description = "凸帕累托前沿，连续问题"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            f1 = x[0]
            g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.array([f1, f2])
        else:
            f1 = x[:, 0]
            g = 1 + 9 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.column_stack([f1, f2])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT2(ZDTBase):
    """ZDT2 - 凹帕累托前沿"""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var, "ZDT2")
        self.description = "凹帕累托前沿，连续问题"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            f1 = x[0]
            g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
            f2 = g * (1 - (f1 / g) ** 2)
            return np.array([f1, f2])
        else:
            f1 = x[:, 0]
            g = 1 + 9 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
            f2 = g * (1 - (f1 / g) ** 2)
            return np.column_stack([f1, f2])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 2
        return np.column_stack([f1, f2])


class ZDT3(ZDTBase):
    """ZDT3 - 断续帕累托前沿"""

    def __init__(self, n_var: int = 30):
        super().__init__(n_var, "ZDT3")
        self.description = "断续帕累托前沿，离散区域"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            f1 = x[0]
            g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
            f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
            return np.array([f1, f2])
        else:
            f1 = x[:, 0]
            g = 1 + 9 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
            f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
            return np.column_stack([f1, f2])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        regions = [
            (0.0, 0.0830015349),
            (0.1822287280, 0.2577623634),
            (0.4093136748, 0.4538821041),
            (0.6183967944, 0.6525117038),
            (0.8233317983, 0.8518328654)
        ]
        points_per_region = n_points // len(regions)
        fronts = []
        for start, end in regions:
            f1 = np.linspace(start, end, points_per_region)
            f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
            fronts.append(np.column_stack([f1, f2]))
        return np.vstack(fronts)


class ZDT4(ZDTBase):
    """ZDT4 - 多峰问题"""

    def __init__(self, n_var: int = 10):
        super().__init__(n_var, "ZDT4")
        self.description = "多峰问题，存在多个局部最优"
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.xl[1:] = -5.0
        self.xu[1:] = 5.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            f1 = x[0]
            g = 1 + 10 * (self.n_var - 1) + np.sum(x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]))
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.array([f1, f2])
        else:
            f1 = x[:, 0]
            g = 1 + 10 * (self.n_var - 1) + np.sum(
                x[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * x[:, 1:]), axis=1
            )
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.column_stack([f1, f2])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT6(ZDTBase):
    """ZDT6 - 非均匀搜索空间"""

    def __init__(self, n_var: int = 10):
        super().__init__(n_var, "ZDT6")
        self.description = "非均匀搜索空间，低密度帕累托前沿"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0])) ** 6
            g = 1 + 9 * (np.sum(x[1:]) / (self.n_var - 1)) ** 0.25
            f2 = g * (1 - (f1 / g) ** 2)
            return np.array([f1, f2])
        else:
            f1 = 1 - np.exp(-4 * x[:, 0]) * (np.sin(6 * np.pi * x[:, 0])) ** 6
            g = 1 + 9 * (np.sum(x[:, 1:], axis=1) / (self.n_var - 1)) ** 0.25
            f2 = g * (1 - (f1 / g) ** 2)
            return np.column_stack([f1, f2])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0.280775, 1, n_points)
        f2 = 1 - f1 ** 2
        return np.column_stack([f1, f2])
