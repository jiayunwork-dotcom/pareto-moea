"""DTLZ系列测试函数"""

import numpy as np
from .base import Problem


class DTLZBase(Problem):
    """DTLZ系列基类"""

    def __init__(self, n_var: int, n_obj: int, name: str = "DTLZ"):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var, n_obj, xl, xu, name)


class DTLZ1(DTLZBase):
    """DTLZ1 - 线性帕累托前沿，多峰"""

    def __init__(self, n_obj: int = 3, k: int = 5):
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, "DTLZ1")
        self.k = k
        self.description = f"{n_obj}目标，线性帕累托前沿，多峰问题"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        n_pop = x.shape[0]
        n_obj = self.n_obj
        k = self.k

        x_m = x[:, n_obj - 1:]
        g = 100 * (k + np.sum((x_m - 0.5) ** 2 - np.cos(20 * np.pi * (x_m - 0.5)), axis=1))

        f = np.zeros((n_pop, n_obj))
        f[:, 0] = 0.5 * np.prod(x[:, :n_obj - 1], axis=1) * (1 + g)
        for i in range(1, n_obj - 1):
            f[:, i] = 0.5 * np.prod(x[:, :n_obj - 1 - i], axis=1) * (1 - x[:, n_obj - 1 - i]) * (1 + g)
        f[:, n_obj - 1] = 0.5 * (1 - x[:, 0]) * (1 + g)

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(self.n_obj, n_points)
        return 0.5 * ref_points


class DTLZ2(DTLZBase):
    """DTLZ2 - 球形帕累托前沿"""

    def __init__(self, n_obj: int = 3, k: int = 10):
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, "DTLZ2")
        self.k = k
        self.description = f"{n_obj}目标，球形帕累托前沿"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        n_pop = x.shape[0]
        n_obj = self.n_obj
        k = self.k

        x_m = x[:, n_obj - 1:]
        g = np.sum((x_m - 0.5) ** 2, axis=1)

        f = np.zeros((n_pop, n_obj))
        f[:, 0] = (1 + g) * np.prod(np.cos(x[:, :n_obj - 1] * np.pi / 2), axis=1)
        for i in range(1, n_obj - 1):
            f[:, i] = (1 + g) * np.prod(np.cos(x[:, :n_obj - 1 - i] * np.pi / 2), axis=1) * np.sin(x[:, n_obj - 1 - i] * np.pi / 2)
        f[:, n_obj - 1] = (1 + g) * np.sin(x[:, 0] * np.pi / 2)

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(self.n_obj, n_points)
        norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
        return ref_points / norms


class DTLZ3(DTLZBase):
    """DTLZ3 - 球形帕累托前沿，多峰"""

    def __init__(self, n_obj: int = 3, k: int = 10):
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, "DTLZ3")
        self.k = k
        self.description = f"{n_obj}目标，球形帕累托前沿，多峰问题"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        n_pop = x.shape[0]
        n_obj = self.n_obj
        k = self.k

        x_m = x[:, n_obj - 1:]
        g = 100 * (k + np.sum((x_m - 0.5) ** 2 - np.cos(20 * np.pi * (x_m - 0.5)), axis=1))

        f = np.zeros((n_pop, n_obj))
        f[:, 0] = (1 + g) * np.prod(np.cos(x[:, :n_obj - 1] * np.pi / 2), axis=1)
        for i in range(1, n_obj - 1):
            f[:, i] = (1 + g) * np.prod(np.cos(x[:, :n_obj - 1 - i] * np.pi / 2), axis=1) * np.sin(x[:, n_obj - 1 - i] * np.pi / 2)
        f[:, n_obj - 1] = (1 + g) * np.sin(x[:, 0] * np.pi / 2)

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(self.n_obj, n_points)
        norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
        return ref_points / norms


class DTLZ4(DTLZBase):
    """DTLZ4 - 球形帕累托前沿，非均匀映射"""

    def __init__(self, n_obj: int = 3, k: int = 10, alpha: float = 100):
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, "DTLZ4")
        self.k = k
        self.alpha = alpha
        self.description = f"{n_obj}目标，球形帕累托前沿，非均匀分布"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        n_pop = x.shape[0]
        n_obj = self.n_obj
        k = self.k
        alpha = self.alpha

        x_m = x[:, n_obj - 1:]
        g = np.sum((x_m - 0.5) ** 2, axis=1)

        x_alpha = x[:, :n_obj - 1] ** alpha

        f = np.zeros((n_pop, n_obj))
        f[:, 0] = (1 + g) * np.prod(np.cos(x_alpha * np.pi / 2), axis=1)
        for i in range(1, n_obj - 1):
            f[:, i] = (1 + g) * np.prod(np.cos(x_alpha[:, :n_obj - 1 - i] * np.pi / 2), axis=1) * np.sin(x_alpha[:, n_obj - 1 - i] * np.pi / 2)
        f[:, n_obj - 1] = (1 + g) * np.sin(x_alpha[:, 0] * np.pi / 2)

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(self.n_obj, n_points)
        norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
        return ref_points / norms


class DTLZ5(DTLZBase):
    """DTLZ5 - 退化帕累托前沿"""

    def __init__(self, n_obj: int = 3, k: int = 10):
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, "DTLZ5")
        self.k = k
        self.description = f"{n_obj}目标，退化帕累托前沿（曲线）"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        n_pop = x.shape[0]
        n_obj = self.n_obj
        k = self.k

        x_m = x[:, n_obj - 1:]
        g = np.sum((x_m - 0.5) ** 2, axis=1)

        theta = np.zeros((n_pop, n_obj - 1))
        theta[:, 0] = x[:, 0] * np.pi / 2
        for i in range(1, n_obj - 1):
            theta[:, i] = (1 / (2 * (1 + g))) * (1 + 2 * g * x[:, i]) * np.pi / 2

        f = np.zeros((n_pop, n_obj))
        f[:, 0] = (1 + g) * np.prod(np.cos(theta), axis=1)
        for i in range(1, n_obj - 1):
            f[:, i] = (1 + g) * np.prod(np.cos(theta[:, :n_obj - 1 - i]), axis=1) * np.sin(theta[:, n_obj - 1 - i])
        f[:, n_obj - 1] = (1 + g) * np.sin(theta[:, 0])

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        n_obj = self.n_obj
        t = np.linspace(0, np.pi / 2, n_points)
        front = np.zeros((n_points, n_obj))
        front[:, 0] = np.cos(t)
        for i in range(1, n_obj - 1):
            front[:, i] = np.cos(t) * np.sin(t)
        front[:, n_obj - 1] = np.sin(t)
        return front


class DTLZ6(DTLZBase):
    """DTLZ6 - 退化帕累托前沿，非均匀"""

    def __init__(self, n_obj: int = 3, k: int = 10):
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, "DTLZ6")
        self.k = k
        self.description = f"{n_obj}目标，退化帕累托前沿，非均匀分布"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        n_pop = x.shape[0]
        n_obj = self.n_obj
        k = self.k

        x_m = x[:, n_obj - 1:]
        g = np.sum(x_m ** 0.1, axis=1)

        theta = np.zeros((n_pop, n_obj - 1))
        theta[:, 0] = x[:, 0] * np.pi / 2
        for i in range(1, n_obj - 1):
            theta[:, i] = (1 / (2 * (1 + g))) * (1 + 2 * g * x[:, i]) * np.pi / 2

        f = np.zeros((n_pop, n_obj))
        f[:, 0] = (1 + g) * np.prod(np.cos(theta), axis=1)
        for i in range(1, n_obj - 1):
            f[:, i] = (1 + g) * np.prod(np.cos(theta[:, :n_obj - 1 - i]), axis=1) * np.sin(theta[:, n_obj - 1 - i])
        f[:, n_obj - 1] = (1 + g) * np.sin(theta[:, 0])

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        n_obj = self.n_obj
        t = np.linspace(0, np.pi / 2, n_points)
        front = np.zeros((n_points, n_obj))
        front[:, 0] = np.cos(t)
        for i in range(1, n_obj - 1):
            front[:, i] = np.cos(t) * np.sin(t)
        front[:, n_obj - 1] = np.sin(t)
        return front


class DTLZ7(DTLZBase):
    """DTLZ7 - 断续帕累托前沿"""

    def __init__(self, n_obj: int = 3, k: int = 20):
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, "DTLZ7")
        self.k = k
        self.description = f"{n_obj}目标，断续帕累托前沿，{2**(n_obj-1)}个不连通区域"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        n_pop = x.shape[0]
        n_obj = self.n_obj
        k = self.k

        f = np.zeros((n_pop, n_obj))
        f[:, :n_obj - 1] = x[:, :n_obj - 1]

        x_m = x[:, n_obj - 1:]
        g = 1 + 9 / k * np.sum(x_m, axis=1)

        h = n_obj
        for i in range(n_obj - 1):
            h -= f[:, i] / (1 + g[:, np.newaxis])[:, 0] * (1 + np.sin(3 * np.pi * f[:, i]))

        f[:, n_obj - 1] = (1 + g) * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> Optional[np.ndarray]:
        return None
