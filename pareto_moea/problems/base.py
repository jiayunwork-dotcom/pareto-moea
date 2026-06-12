"""问题基类定义"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, List


class Problem(ABC):
    """优化问题基类"""

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        xl: Optional[np.ndarray] = None,
        xu: Optional[np.ndarray] = None,
        name: str = "Problem",
        description: str = ""
    ):
        self.n_var = n_var
        self.n_obj = n_obj
        self.name = name
        self.description = description

        if xl is None:
            xl = np.zeros(n_var)
        if xu is None:
            xu = np.ones(n_var)

        self.xl = np.asarray(xl, dtype=float)
        self.xu = np.asarray(xu, dtype=float)

        if self.xl.shape != (n_var,) or self.xu.shape != (n_var,):
            raise ValueError("Variable bounds must have shape (n_var,)")

        self.n_constr = 0
        self.constraints: List[Callable] = []

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """计算目标函数值

        Args:
            x: 决策变量，形状为 (n_var,) 或 (n_pop, n_var)

        Returns:
            目标函数值，形状为 (n_obj,) 或 (n_pop, n_obj)
        """
        pass

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """计算约束违反量（g(x) <= 0 形式，正值表示违反）

        Args:
            x: 决策变量，形状为 (n_var,) 或 (n_pop, n_var)

        Returns:
            约束违反量，形状为 (n_constr,) 或 (n_pop, n_constr)
        """
        if self.n_constr == 0:
            if x.ndim == 1:
                return np.array([])
            else:
                return np.zeros((len(x), 0))

        if x.ndim == 1:
            violations = []
            for constr in self.constraints:
                val = constr(x)
                violations.append(max(0.0, val))
            return np.array(violations)
        else:
            violations = []
            for constr in self.constraints:
                vals = np.array([constr(xi) for xi in x])
                violations.append(np.maximum(0.0, vals))
            return np.column_stack(violations)

    def constraint_violation(self, x: np.ndarray) -> np.ndarray:
        """计算总约束违反量（所有约束违反量之和）

        Args:
            x: 决策变量，形状为 (n_var,) 或 (n_pop, n_var)

        Returns:
            总约束违反量，标量或形状为 (n_pop,)
        """
        cv = self.evaluate_constraints(x)
        if cv.size == 0:
            if x.ndim == 1:
                return 0.0
            else:
                return np.zeros(len(x))
        return np.sum(cv, axis=-1)

    def is_feasible(self, x: np.ndarray) -> np.ndarray:
        """判断解是否可行

        Args:
            x: 决策变量，形状为 (n_var,) 或 (n_pop, n_var)

        Returns:
            布尔值，表示是否可行
        """
        return self.constraint_violation(x) <= 1e-10

    def pareto_front(self, n_points: int = 1000) -> Optional[np.ndarray]:
        """生成真实帕累托前沿（如果已知）

        Args:
            n_points: 前沿上的点数

        Returns:
            帕累托前沿点，形状为 (n_points, n_obj)，如果未知则返回None
        """
        return None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def __repr__(self):
        return f"{self.name}(n_var={self.n_var}, n_obj={self.n_obj}, n_constr={self.n_constr})"


class CustomProblem(Problem):
    """用户自定义问题"""

    def __init__(
        self,
        objective_func: Callable,
        n_var: int,
        n_obj: int,
        xl: Optional[np.ndarray] = None,
        xu: Optional[np.ndarray] = None,
        constraints: Optional[List[Callable]] = None,
        name: str = "CustomProblem",
        description: str = ""
    ):
        super().__init__(n_var, n_obj, xl, xu, name, description)
        self.objective_func = objective_func

        if constraints is not None:
            self.constraints = constraints
            self.n_constr = len(constraints)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return np.asarray(self.objective_func(x), dtype=float)
        else:
            results = [self.objective_func(xi) for xi in x]
            return np.asarray(results, dtype=float)
