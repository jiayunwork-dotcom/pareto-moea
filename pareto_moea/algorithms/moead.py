"""MOEA/D 算法实现"""

import numpy as np
from typing import Optional
from .base import Algorithm


class MOEAD(Algorithm):
    """基于分解的多目标进化算法 (MOEA/D)

    使用切比雪夫聚合函数将多目标问题分解为多个单目标子问题，
    每个子问题对应一个权重向量，利用邻域信息进行进化。
    """

    def __init__(
        self,
        pop_size: int = 100,
        n_gen: int = 100,
        n_weights: int = 100,
        neighbor_size: int = 20,
        crossover_prob: float = 0.9,
        crossover_eta: float = 20.0,
        mutation_prob: Optional[float] = None,
        mutation_eta: float = 20.0,
        constraint_strategy: str = "feasibility_rule",
        penalty_factor: float = 100.0,
        seed: Optional[int] = None
    ):
        super().__init__(
            pop_size=pop_size,
            n_gen=n_gen,
            crossover_prob=crossover_prob,
            crossover_eta=crossover_eta,
            mutation_prob=mutation_prob,
            mutation_eta=mutation_eta,
            constraint_strategy=constraint_strategy,
            penalty_factor=penalty_factor,
            seed=seed
        )
        self.n_weights = n_weights
        self.neighbor_size = neighbor_size

        self.weights = None
        self.neighbors = None
        self.z = None
        self.population = None
        self.objectives = None
        self.cv = None

    def setup(self, problem):
        """设置优化问题并初始化MOEA/D组件"""
        super().setup(problem)
        self._initialize_weights()
        self.pop_size = self.n_weights
        self._initialize_neighbors()

    def _initialize_weights(self):
        """生成均匀分布的权重向量"""
        from ..utils.pareto_utils import uniform_reference_points

        n_obj = self.problem.n_obj

        if n_obj == 1:
            self.weights = np.ones((self.n_weights, 1))
            return

        n_divisions = self._find_n_divisions(n_obj, self.n_weights)
        self.weights = uniform_reference_points(n_obj, n_divisions)

        if len(self.weights) < self.n_weights:
            extra = self.n_weights - len(self.weights)
            extra_weights = self.rng.dirichlet(np.ones(n_obj), size=extra)
            self.weights = np.vstack([self.weights, extra_weights])
        elif len(self.weights) > self.n_weights:
            indices = self.rng.choice(len(self.weights), self.n_weights, replace=False)
            self.weights = self.weights[indices]

        self.n_weights = len(self.weights)

    def _find_n_divisions(self, n_obj: int, target: int) -> int:
        """查找最接近目标点数的等分数"""
        best_h = 1
        best_diff = float('inf')

        for h in range(1, 200):
            n = self._comb(h + n_obj - 1, n_obj - 1)
            diff = abs(n - target)
            if diff < best_diff:
                best_diff = diff
                best_h = h
            if n > target and diff > best_diff:
                break

        return best_h

    def _comb(self, n: int, k: int) -> int:
        """计算组合数 C(n, k)"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(1, k + 1):
            result = result * (n - k + i) // i
        return result

    def _initialize_neighbors(self):
        """计算每个权重向量的邻域（基于欧氏距离）"""
        n = len(self.weights)
        self.neighbors = np.zeros((n, self.neighbor_size), dtype=int)

        for i in range(n):
            distances = np.sum((self.weights - self.weights[i]) ** 2, axis=1)
            sorted_indices = np.argsort(distances)
            self.neighbors[i] = sorted_indices[:self.neighbor_size]

    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        pop = self._initialize_random()
        self.population = pop
        self.objectives = self.problem.evaluate(pop)
        self.cv = self.problem.constraint_violation(pop)
        self._initialize_ideal_point()
        return pop

    def _initialize_ideal_point(self):
        """初始化理想点（每个目标的最小值）"""
        self.z = np.min(self.objectives, axis=0)

    def _update_ideal_point(self, objectives: np.ndarray):
        """更新理想点"""
        self.z = np.minimum(self.z, np.min(objectives, axis=0))

    def _chebyshev(self, objective: np.ndarray, weight: np.ndarray) -> float:
        """切比雪夫聚合函数

        Args:
            objective: 目标函数值，形状为 (n_obj,)
            weight: 权重向量，形状为 (n_obj,)

        Returns:
            切比雪夫标量值
        """
        weight = np.where(weight == 0, 1e-10, weight)
        return np.max(weight * np.abs(objective - self.z))

    def _tchebyshev_fitness(self, objectives: np.ndarray,
                            weights: np.ndarray) -> np.ndarray:
        """计算种群在所有权重向量下的切比雪夫适应度

        Args:
            objectives: 目标函数值，形状为 (n_pop, n_obj)
            weights: 权重向量，形状为 (n_weights, n_obj)

        Returns:
            适应度矩阵，形状为 (n_pop, n_weights)
        """
        n_pop = len(objectives)
        n_weights = len(weights)
        fitness = np.zeros((n_pop, n_weights))

        for i in range(n_pop):
            for j in range(n_weights):
                fitness[i, j] = self._chebyshev(objectives[i], weights[j])

        return fitness

    def _evolve(self, population: np.ndarray, objectives: np.ndarray,
                cv: np.ndarray, gen: int) -> tuple:
        """执行一代进化

        Args:
            population: 当前种群，形状为 (pop_size, n_var)
            objectives: 当前目标值，形状为 (pop_size, n_obj)
            cv: 当前约束违反量，形状为 (pop_size,)
            gen: 当前代数

        Returns:
            (new_population, new_objectives, new_cv)
        """
        self.population = population.copy()
        self.objectives = objectives.copy()
        self.cv = cv.copy()

        n_weights = self.n_weights
        n_var = self.problem.n_var

        for i in range(n_weights):
            neighbor_indices = self.neighbors[i]

            parent1_idx = self.rng.choice(neighbor_indices)
            parent2_idx = self.rng.choice(neighbor_indices)
            while parent2_idx == parent1_idx:
                parent2_idx = self.rng.choice(neighbor_indices)

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            child1, child2 = self._sbx_crossover(parent1, parent2)

            child1 = self._polynomial_mutation(child1)
            child2 = self._polynomial_mutation(child2)

            child1_obj = self.problem.evaluate(child1)
            child2_obj = self.problem.evaluate(child2)
            child1_cv = self.problem.constraint_violation(child1)
            child2_cv = self.problem.constraint_violation(child2)

            self._update_ideal_point(np.vstack([child1_obj, child2_obj]))

            weight_i = self.weights[i]

            for j in neighbor_indices:
                weight_j = self.weights[j]

                current_fitness = self._chebyshev(self.objectives[j], weight_j)

                child1_fit = self._chebyshev(child1_obj, weight_j)
                child2_fit = self._chebyshev(child2_obj, weight_j)

                if self.problem.n_constr > 0:
                    current_cv = self.cv[j]

                    if child1_cv <= 1e-10 and current_cv > 1e-10:
                        child1_better = True
                    elif child1_cv > 1e-10 and current_cv <= 1e-10:
                        child1_better = False
                    elif child1_cv > 1e-10 and current_cv > 1e-10:
                        child1_better = child1_cv < current_cv
                    else:
                        child1_better = child1_fit < current_fitness

                    if child2_cv <= 1e-10 and current_cv > 1e-10:
                        child2_better = True
                    elif child2_cv > 1e-10 and current_cv <= 1e-10:
                        child2_better = False
                    elif child2_cv > 1e-10 and current_cv > 1e-10:
                        child2_better = child2_cv < current_cv
                    else:
                        child2_better = child2_fit < current_fitness
                else:
                    child1_better = child1_fit < current_fitness
                    child2_better = child2_fit < current_fitness

                if child1_better:
                    self.population[j] = child1.copy()
                    self.objectives[j] = child1_obj.copy()
                    self.cv[j] = child1_cv
                    current_fitness = child1_fit

                if child2_better:
                    self.population[j] = child2.copy()
                    self.objectives[j] = child2_obj.copy()
                    self.cv[j] = child2_cv

        return self.population.copy(), self.objectives.copy(), self.cv.copy()

    def get_params(self) -> dict:
        """获取算法参数"""
        params = super().get_params()
        params.update({
            'n_weights': self.n_weights,
            'neighbor_size': self.neighbor_size
        })
        return params
