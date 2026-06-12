"""SMS-EMOA算法实现"""

import numpy as np
from .base import Algorithm
from ..utils.pareto_utils import fast_non_dominated_sort
from ..metrics.indicators import hypervolume


class SMSEMOA(Algorithm):
    """基于超体积贡献的稳态多目标进化算法
    (S-Metric Selection Evolutionary Multi-Objective Algorithm)

    核心特点：
    - 基于超体积贡献的稳态选择
    - 每次只淘汰超体积贡献最小的个体
    - 稳态进化：每代只产生少量子代（通常1-2个）

    算法流程：
    1. 初始化种群
    2. 对种群进行非支配排序
    3. 选择最后一个前沿
    4. 计算该前沿中每个个体的超体积贡献
    5. 移除超体积贡献最小的个体，保持种群大小不变
    6. 选择、交叉、变异产生新个体加入种群
    7. 重复步骤2-6直到达到最大代数
    """

    def __init__(
        self,
        pop_size: int = 100,
        n_gen: int = 100,
        crossover_prob: float = 0.9,
        crossover_eta: float = 20.0,
        mutation_prob: float = None,
        mutation_eta: float = 20.0,
        constraint_strategy: str = "feasibility_rule",
        penalty_factor: float = 100.0,
        seed: int = None,
        reference_point: np.ndarray = None,
        n_offspring: int = 1
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
        self.reference_point = reference_point
        self.n_offspring = n_offspring
        self._ref_point = None

    def setup(self, problem):
        """设置优化问题"""
        super().setup(problem)
        if self.reference_point is not None:
            self._ref_point = np.asarray(self.reference_point, dtype=float)

    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        return self._initialize_random()

    def _evolve(self, population: np.ndarray, objectives: np.ndarray,
                cv: np.ndarray, gen: int) -> tuple:
        """执行一代进化

        Args:
            population: 当前种群，形状为 (pop_size, n_var)
            objectives: 目标函数值，形状为 (pop_size, n_obj)
            cv: 约束违反值，形状为 (pop_size,)
            gen: 当前代数

        Returns:
            (new_population, new_objectives, new_cv)
        """
        fitness = self._evaluate_fitness(objectives, cv)

        if self._ref_point is None:
            self._ref_point = self._compute_reference_point(objectives)

        for _ in range(self.n_offspring):
            parent1, parent2 = self._selection(population, fitness)

            child1, child2 = self._sbx_crossover(parent1, parent2)

            if self.rng.rand() < 0.5:
                child = self._polynomial_mutation(child1)
            else:
                child = self._polynomial_mutation(child2)

            child_obj = self.problem.evaluate(child.reshape(1, -1))[0]
            child_cv = self.problem.constraint_violation(child.reshape(1, -1))

            population = np.vstack([population, child.reshape(1, -1)])
            objectives = np.vstack([objectives, child_obj.reshape(1, -1)])
            cv = np.concatenate([cv, child_cv])

            fitness = self._evaluate_fitness(objectives, cv)

            if len(population) > self.pop_size:
                remove_idx = self._select_worst_individual(fitness, objectives)
                population = np.delete(population, remove_idx, axis=0)
                objectives = np.delete(objectives, remove_idx, axis=0)
                cv = np.delete(cv, remove_idx)

        return population, objectives, cv

    def _compute_reference_point(self, objectives: np.ndarray) -> np.ndarray:
        """自动计算参考点

        参考点为各目标最大值的1.1倍。

        Args:
            objectives: 目标函数值，形状为 (n_pop, n_obj)

        Returns:
            参考点，形状为 (n_obj,)
        """
        max_obj = np.max(objectives, axis=0)
        ref_point = max_obj * 1.1
        return ref_point

    def _selection(self, population: np.ndarray,
                   fitness: np.ndarray) -> tuple:
        """二元锦标赛选择

        基于非支配排序等级进行选择，同一等级随机选择。

        Args:
            population: 种群，形状为 (pop_size, n_var)
            fitness: 适应度值（目标函数或约束处理后的值）

        Returns:
            (parent1, parent2) 两个父代个体
        """
        ranks = fast_non_dominated_sort(fitness)

        idx1 = self._binary_tournament(ranks)
        idx2 = self._binary_tournament(ranks)

        while idx2 == idx1:
            idx2 = self._binary_tournament(ranks)

        return population[idx1].copy(), population[idx2].copy()

    def _binary_tournament(self, ranks: np.ndarray) -> int:
        """二元锦标赛选择（返回索引）

        Args:
            ranks: 每个个体的非支配排序等级

        Returns:
            选中个体的索引
        """
        n_pop = len(ranks)
        indices = self.rng.choice(n_pop, size=2, replace=False)

        idx1, idx2 = indices[0], indices[1]

        if ranks[idx1] < ranks[idx2]:
            return idx1
        elif ranks[idx1] > ranks[idx2]:
            return idx2
        else:
            return idx1 if self.rng.rand() < 0.5 else idx2

    def _select_worst_individual(self, fitness: np.ndarray,
                                 objectives: np.ndarray) -> int:
        """选择超体积贡献最小的个体进行淘汰

        基于 fitness 进行非支配排序确定最后一个前沿，
        基于 objectives 计算该前沿内个体的超体积贡献。

        Args:
            fitness: 适应度值（考虑约束），形状为 (n_pop, n_obj)
            objectives: 目标函数值，形状为 (n_pop, n_obj)

        Returns:
            要淘汰的个体索引
        """
        ranks = fast_non_dominated_sort(fitness)
        max_rank = np.max(ranks)

        last_front_indices = np.where(ranks == max_rank)[0]

        if len(last_front_indices) == 1:
            return last_front_indices[0]

        front_obj = objectives[last_front_indices]

        if front_obj.shape[1] == 2:
            contributions = self._hv_contribution_2d(
                front_obj, self._ref_point
            )
        else:
            contributions = self._hv_contribution_exclusion(
                front_obj, self._ref_point
            )

        min_contrib_idx = np.argmin(contributions)
        return last_front_indices[min_contrib_idx]

    def _hv_contribution_2d(self, front: np.ndarray,
                            ref_point: np.ndarray) -> np.ndarray:
        """2目标超体积贡献计算（高效方法）

        对于最小化问题，按f1升序排列前沿点后，每个点的超体积贡献为：
        - 左端点：(f1_1 - f1_0) * (ref_y - f2_0)
        - 右端点：(ref_x - f1_{n-1}) * (f2_{n-2} - f2_{n-1})
        - 内部点：(f1_{i+1} - f1_i) * (f2_{i-1} - f2_i)

        Args:
            front: 同一前沿的目标值，形状为 (n, 2)
            ref_point: 参考点，形状为 (2,)

        Returns:
            每个个体的超体积贡献，形状为 (n,)
        """
        n = len(front)
        contributions = np.zeros(n)

        if n == 1:
            width = ref_point[0] - front[0, 0]
            height = ref_point[1] - front[0, 1]
            contributions[0] = width * height
            return contributions

        sorted_idx = np.argsort(front[:, 0])
        sorted_front = front[sorted_idx]

        for i in range(n):
            if i == 0:
                width = sorted_front[i + 1, 0] - sorted_front[i, 0]
                height = ref_point[1] - sorted_front[i, 1]
            elif i == n - 1:
                width = ref_point[0] - sorted_front[i, 0]
                height = sorted_front[i - 1, 1] - sorted_front[i, 1]
            else:
                width = sorted_front[i + 1, 0] - sorted_front[i, 0]
                height = sorted_front[i - 1, 1] - sorted_front[i, 1]

            contributions[sorted_idx[i]] = width * height

        return contributions

    def _hv_contribution_exclusion(self, front: np.ndarray,
                                   ref_point: np.ndarray) -> np.ndarray:
        """多目标超体积贡献计算（排除法）

        计算移除每个点后的超体积变化。

        Args:
            front: 同一前沿的目标值，形状为 (n, n_obj)
            ref_point: 参考点，形状为 (n_obj,)

        Returns:
            每个个体的超体积贡献，形状为 (n,)
        """
        n = len(front)
        contributions = np.zeros(n)

        total_hv = hypervolume(front, ref_point)

        for i in range(n):
            remaining = np.delete(front, i, axis=0)
            remaining_hv = hypervolume(remaining, ref_point)
            contributions[i] = total_hv - remaining_hv

        return contributions

    def get_params(self) -> dict:
        """获取算法参数"""
        params = super().get_params()
        params.update({
            'reference_point': self.reference_point,
            'n_offspring': self.n_offspring
        })
        return params
