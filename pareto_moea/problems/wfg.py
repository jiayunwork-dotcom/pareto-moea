"""WFG系列测试函数"""

import numpy as np
from .base import Problem


class WFGBase(Problem):
    """WFG系列基类"""

    def __init__(self, n_var: int, n_obj: int, k: int, name: str = "WFG"):
        self.k = k
        xl = np.zeros(n_var)
        xu = np.array([2.0 * (i + 1) for i in range(n_var)])
        super().__init__(n_var, n_obj, xl, xu, name)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return x / self.xu

    def _s_linear(self, y: np.ndarray, a: float) -> np.ndarray:
        return np.abs(y - a) / np.abs(np.floor(a - y) + a)

    def _s_flat(self, y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        tmp1 = np.minimum(0.0, np.floor(y - b)) * a * (b - y) / b
        tmp2 = np.minimum(0.0, np.floor(c - y)) * (1 - a) * (y - c) / (1 - c)
        return a + tmp1 + tmp2

    def _s_multi(self, y: np.ndarray, a: int, b: float, c: float) -> np.ndarray:
        tmp1 = np.abs(y - c) / (2 * (np.floor(c - y) + c))
        tmp2 = (4 * a + 2) * np.pi * (0.5 - tmp1)
        return (1 + np.cos(tmp2) + 4 * b * tmp1 ** 2) / (4 * b + 2)

    def _s_decept(self, y: np.ndarray, a: float = 0.35, b: float = 0.001, c: float = 0.05) -> np.ndarray:
        tmp1 = np.floor(y - a + b) * (1 - c + (a - b) / b)
        tmp2 = np.floor(a + b - y) * (1 - c + (1 - a - b) / b)
        return 1 + (np.abs(y - a) - b) * (tmp1 / (a - b) + tmp2 / (1 - a - b) + 1 / b)

    def _s_param(self, y: np.ndarray, u: float) -> np.ndarray:
        return np.power(y, np.log2(u))

    def _r_sum(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        w_sum = np.sum(w)
        return np.sum(y * w, axis=-1) / (w_sum * np.max(y, axis=-1) + 1e-10)

    def _r_nonsep(self, y: np.ndarray, a: int) -> np.ndarray:
        n = y.shape[-1]
        result = np.zeros(y.shape[:-1])
        y_padded = np.concatenate([y, y[..., :a-1]], axis=-1)
        for j in range(n):
            result += y[..., j]
            for k in range(1, a):
                result += np.abs(y[..., j] - y[..., (j + k) % n])
        denominator = n * a * (1.0 + 2.0 * a) / (2.0 * a)
        return result / denominator

    def _shape_linear(self, x: np.ndarray, m: int) -> np.ndarray:
        M = x.shape[-1] + 1
        if m == M:
            return 1 - x[..., 0]
        elif m == 1:
            return np.prod(x, axis=-1)
        else:
            return np.prod(x[..., :M - m], axis=-1) * (1 - x[..., M - m])

    def _shape_convex(self, x: np.ndarray, m: int) -> np.ndarray:
        M = x.shape[-1] + 1
        if m == M:
            return 1 - np.sin(x[..., 0] * np.pi / 2)
        elif m == 1:
            return np.prod(1 - np.cos(x * np.pi / 2), axis=-1)
        else:
            return np.prod(1 - np.cos(x[..., :M - m] * np.pi / 2), axis=-1) * \
                   (1 - np.sin(x[..., M - m] * np.pi / 2))

    def _shape_concave(self, x: np.ndarray, m: int) -> np.ndarray:
        M = x.shape[-1] + 1
        if m == M:
            return np.cos(x[..., 0] * np.pi / 2)
        elif m == 1:
            return np.prod(np.sin(x * np.pi / 2), axis=-1)
        else:
            return np.prod(np.sin(x[..., :M - m] * np.pi / 2), axis=-1) * \
                   np.cos(x[..., M - m] * np.pi / 2)

    def _shape_mixed(self, x: np.ndarray, a: int, alpha: float) -> np.ndarray:
        tmp = 2 * a * np.pi
        return 1 - x[..., 0] * np.power(
            np.abs(np.cos(tmp * x[..., 1] + np.pi / 2)) / tmp + np.cos(tmp * x[..., 1]),
            alpha
        )

    def _shape_disc(self, x: np.ndarray, a: int, b: float, c: float) -> np.ndarray:
        tmp1 = np.abs(np.cos(a * np.pi * x[..., 1] + np.pi / 2)) / (a * np.pi)
        return 1 - x[..., 0] * np.power(
            1 - x[..., 1] / (c - np.floor(c - x[..., 1])) * (b - tmp1),
            3
        )

    def _calculate_f(self, x_m: np.ndarray, h: np.ndarray, s: np.ndarray, d: float = 1.0) -> np.ndarray:
        return d * x_m[..., np.newaxis] + s * h


class WFG1(WFGBase):
    """WFG1 - 凸帕累托前沿，可分解，多峰"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG1")
        self.description = f"{n_obj}目标，凸帕累托前沿，可分解，多峰"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        t1[:, :k] = y[:, :k]
        for i in range(k, self.n_var):
            t1[:, i] = self._s_linear(y[:, i], 0.35)

        t2 = np.zeros_like(y)
        t2[:, :k] = t1[:, :k]
        for i in range(k, self.n_var):
            t2[:, i] = self._s_flat(t1[:, i], 0.8, 0.75, 0.85)

        t3 = np.zeros((n_pop, n_obj))
        t3[:, :n_obj - 1] = t2[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t3[:, n_obj - 1] = self._r_sum(t2[:, k:], w)

        t4 = np.zeros_like(t3)
        for i in range(n_obj - 1):
            t4[:, i] = self._s_multi(t3[:, i], 30, 95, 0.35)
        t4[:, n_obj - 1] = self._s_linear(t3[:, n_obj - 1], 0.3)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_convex(t4[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t4[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(n_obj, n_points)
        return s * ref_points


class WFG2(WFGBase):
    """WFG2 - 凸帕累托前沿，不可分解，多峰"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG2")
        self.description = f"{n_obj}目标，凸帕累托前沿，不可分解，多峰"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        t1[:, :k] = y[:, :k]
        for i in range(k, self.n_var):
            t1[:, i] = self._s_linear(y[:, i], 0.35)

        t2 = np.zeros_like(y)
        t2[:, :k] = t1[:, :k]
        n_pairs = l // 2
        for i in range(n_pairs):
            idx = k + 2 * i
            t2[:, idx:idx + 2] = self._r_nonsep(t1[:, idx:idx + 2], 2)
        if l % 2 == 1:
            t2[:, -1] = t1[:, -1]

        t3 = np.zeros((n_pop, n_obj))
        t3[:, :n_obj - 1] = t2[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t3[:, n_obj - 1] = self._r_sum(t2[:, k:], w)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_convex(t3[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t3[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(n_obj, n_points)
        return s * ref_points


class WFG3(WFGBase):
    """WFG3 - 线性帕累托前沿，可分解，多峰（退化）"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG3")
        self.description = f"{n_obj}目标，线性帕累托前沿，可分解，多峰（退化）"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        t1[:, :k] = y[:, :k]
        for i in range(k, self.n_var):
            t1[:, i] = self._s_linear(y[:, i], 0.35)

        t2 = np.zeros_like(y)
        t2[:, :k] = t1[:, :k]
        n_pairs = l // 2
        for i in range(n_pairs):
            idx = k + 2 * i
            t2[:, idx:idx + 2] = self._r_nonsep(t1[:, idx:idx + 2], 2)
        if l % 2 == 1:
            t2[:, -1] = t1[:, -1]

        t3 = np.zeros((n_pop, n_obj))
        t3[:, :n_obj - 1] = t2[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t3[:, n_obj - 1] = self._r_sum(t2[:, k:], w)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_linear(t3[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t3[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        t = np.linspace(0, 1, n_points)
        front = np.zeros((n_points, n_obj))
        front[:, 0] = s[0] * t
        front[:, n_obj - 1] = s[n_obj - 1] * (1 - t)
        for i in range(1, n_obj - 1):
            front[:, i] = s[i] * 0.5 * (1 - t)
        return front


class WFG4(WFGBase):
    """WFG4 - 凹帕累托前沿，可分解，单峰"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG4")
        self.description = f"{n_obj}目标，凹帕累托前沿，可分解，单峰"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        for i in range(self.n_var):
            t1[:, i] = self._s_multi(y[:, i], 30, 95, 0.35)

        t2 = np.zeros((n_pop, n_obj))
        t2[:, :n_obj - 1] = t1[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t2[:, n_obj - 1] = self._r_sum(t1[:, k:], w)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_concave(t2[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t2[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(n_obj, n_points)
        return s * ref_points


class WFG5(WFGBase):
    """WFG5 - 凹帕累托前沿，可分解，欺骗性"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG5")
        self.description = f"{n_obj}目标，凹帕累托前沿，可分解，欺骗性"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        for i in range(self.n_var):
            t1[:, i] = self._s_decept(y[:, i])

        t2 = np.zeros((n_pop, n_obj))
        t2[:, :n_obj - 1] = t1[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t2[:, n_obj - 1] = self._r_sum(t1[:, k:], w)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_concave(t2[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t2[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(n_obj, n_points)
        return s * ref_points


class WFG6(WFGBase):
    """WFG6 - 凹帕累托前沿，不可分解，单峰"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG6")
        self.description = f"{n_obj}目标，凹帕累托前沿，不可分解，单峰"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        t1[:, :k] = y[:, :k]
        for i in range(k, self.n_var):
            t1[:, i] = self._s_linear(y[:, i], 0.35)

        t2 = np.zeros_like(y)
        t2[:, :k] = t1[:, :k]
        n_pairs = l // 2
        for i in range(n_pairs):
            idx = k + 2 * i
            t2[:, idx:idx + 2] = self._r_nonsep(t1[:, idx:idx + 2], 2)
        if l % 2 == 1:
            t2[:, -1] = t1[:, -1]

        t3 = np.zeros((n_pop, n_obj))
        t3[:, :n_obj - 1] = t2[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t3[:, n_obj - 1] = self._r_sum(t2[:, k:], w)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_concave(t3[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t3[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(n_obj, n_points)
        return s * ref_points


class WFG7(WFGBase):
    """WFG7 - 凹帕累托前沿，可分解，有偏"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG7")
        self.description = f"{n_obj}目标，凹帕累托前沿，可分解，有偏"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        for i in range(k):
            t1[:, i] = self._s_param(y[:, i], i + 1)
        t1[:, k:] = y[:, k:]

        t2 = np.zeros_like(y)
        t2[:, :k] = t1[:, :k]
        for i in range(k, self.n_var):
            t2[:, i] = self._s_linear(t1[:, i], 0.35)

        t3 = np.zeros((n_pop, n_obj))
        t3[:, :n_obj - 1] = t2[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t3[:, n_obj - 1] = self._r_sum(t2[:, k:], w)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_concave(t3[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t3[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(n_obj, n_points)
        return s * ref_points


class WFG8(WFGBase):
    """WFG8 - 凹帕累托前沿，不可分解，有偏"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG8")
        self.description = f"{n_obj}目标，凹帕累托前沿，不可分解，有偏"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        t1[:, :k] = y[:, :k]
        for i in range(k, self.n_var):
            t1[:, i] = self._s_param(y[:, i], i + 1 - k)

        t2 = np.zeros_like(y)
        t2[:, :k] = t1[:, :k]
        for i in range(k, self.n_var):
            t2[:, i] = self._s_linear(t1[:, i], 0.35)

        t3 = np.zeros_like(y)
        t3[:, :k] = t2[:, :k]
        n_pairs = l // 2
        for i in range(n_pairs):
            idx = k + 2 * i
            t3[:, idx:idx + 2] = self._r_nonsep(t2[:, idx:idx + 2], 2)
        if l % 2 == 1:
            t3[:, -1] = t2[:, -1]

        t4 = np.zeros((n_pop, n_obj))
        t4[:, :n_obj - 1] = t3[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t4[:, n_obj - 1] = self._r_sum(t3[:, k:], w)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_concave(t4[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t4[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(n_obj, n_points)
        return s * ref_points


class WFG9(WFGBase):
    """WFG9 - 凹帕累托前沿，不可分解，欺骗性，有偏"""

    def __init__(self, n_obj: int = 3, k: int = 5, n_var: int = None):
        if n_var is None:
            n_var = k + 10
        super().__init__(n_var, n_obj, k, "WFG9")
        self.description = f"{n_obj}目标，凹帕累托前沿，不可分解，欺骗性，有偏"

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
        l = self.n_var - k

        y = self._normalize(x)

        t1 = np.zeros_like(y)
        for i in range(k):
            t1[:, i] = self._s_param(y[:, i], i + 1)
        for i in range(k, self.n_var):
            t1[:, i] = self._s_decept(y[:, i])

        t2 = np.zeros_like(y)
        t2[:, :k] = t1[:, :k]
        n_pairs = l // 2
        for i in range(n_pairs):
            idx = k + 2 * i
            t2[:, idx:idx + 2] = self._r_nonsep(t1[:, idx:idx + 2], 2)
        if l % 2 == 1:
            t2[:, -1] = t1[:, -1]

        t3 = np.zeros((n_pop, n_obj))
        t3[:, :n_obj - 1] = t2[:, :n_obj - 1]
        w = np.array([2.0 * (i + 1) for i in range(l)])
        t3[:, n_obj - 1] = self._r_sum(t2[:, k:], w)

        h = np.zeros((n_pop, n_obj))
        for i in range(n_obj):
            h[:, i] = self._shape_concave(t3[:, :n_obj - 1], i + 1)

        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        d = 1.0
        f = d * t3[:, n_obj - 1][..., np.newaxis] + s * h

        if single:
            return f[0]
        return f

    def pareto_front(self, n_points: int = 10000) -> np.ndarray:
        n_obj = self.n_obj
        s = np.array([2.0 * (i + 1) for i in range(n_obj)])
        from ..utils.pareto_utils import uniform_reference_points
        ref_points = uniform_reference_points(n_obj, n_points)
        return s * ref_points
