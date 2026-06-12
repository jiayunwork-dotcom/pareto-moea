"""NSGA-II算法实现"""

import numpy as np
from .base import Algorithm
from ..utils.pareto_utils import fast_non_dominated_sort, crowding_distance
from ..utils.operators import sbx_crossover, polynomial_mutation


class NSGA2(Algorithm):
    """非支配排序遗传算法 II (Non-dominated Sorting Genetic Algorithm II)

    核心组件：
    - 快速非支配排序 (Fast Non-dominated Sorting)
    - 拥挤距离 (Crowding Distance)
    - 二元锦标赛选择 (Binary Tournament Selection)
    - SBX交叉 (Simulated Binary Crossover)
    - 多项式变异 (Polynomial Mutation)
    """

    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        return self._initialize_random()

    def _evolve(self, population: np.ndarray, objectives: np.ndarray,
                cv: np.ndarray, gen: int) -> tuple:
        """执行一代进化

        Args:
            population: 当前种群，形状为 (pop_size, n_var)
            objectives: 目标函数值，形状为 (pop_size, n_obj)
            cv: 约束违反值，形状为 (pop_size, n_constr)
            gen: 当前代数

        Returns:
            (new_population, new_objectives, new_cv)
        """
        fitness = self._evaluate_fitness(objectives, cv)

        offspring = self._create_offspring(population, fitness)
        offspring_obj = self.problem.evaluate(offspring)
        offspring_cv = self.problem.constraint_violation(offspring)

        merged_pop = np.vstack([population, offspring])
        merged_obj = np.vstack([objectives, offspring_obj])
        merged_cv = np.concatenate([cv, offspring_cv])

        new_population, new_objectives, new_cv = self._environmental_selection(
            merged_pop, merged_obj, merged_cv
        )

        return new_population, new_objectives, new_cv

    def _create_offspring(self, population: np.ndarray,
                          fitness: np.ndarray) -> np.ndarray:
        """创建子代种群"""
        pop_size = len(population)
        offspring = np.zeros_like(population)

        ranks = fast_non_dominated_sort(fitness)

        for i in range(0, pop_size, 2):
            parent1 = self._binary_tournament(population, ranks, fitness)
            parent2 = self._binary_tournament(population, ranks, fitness)

            child1, child2 = sbx_crossover(
                parent1, parent2,
                self.problem.xl, self.problem.xu,
                self.crossover_eta, self.crossover_prob
            )

            child1 = polynomial_mutation(
                child1,
                self.problem.xl, self.problem.xu,
                self.mutation_eta, self.mutation_prob
            )
            child2 = polynomial_mutation(
                child2,
                self.problem.xl, self.problem.xu,
                self.mutation_eta, self.mutation_prob
            )

            offspring[i] = child1
            if i + 1 < pop_size:
                offspring[i + 1] = child2

        return offspring

    def _binary_tournament(self, population: np.ndarray,
                           ranks: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """基于拥挤距离比较的二元锦标赛选择"""
        n_pop = len(population)
        indices = self.rng.choice(n_pop, size=2, replace=False)

        idx1, idx2 = indices[0], indices[1]

        if ranks[idx1] < ranks[idx2]:
            winner_idx = idx1
        elif ranks[idx1] > ranks[idx2]:
            winner_idx = idx2
        else:
            front_mask = ranks == ranks[idx1]
            front_indices = np.where(front_mask)[0]
            front_fitness = fitness[front_indices]
            cd = crowding_distance(front_fitness)

            pos1 = np.where(front_indices == idx1)[0][0]
            pos2 = np.where(front_indices == idx2)[0][0]

            if cd[pos1] > cd[pos2]:
                winner_idx = idx1
            elif cd[pos1] < cd[pos2]:
                winner_idx = idx2
            else:
                winner_idx = idx1 if self.rng.rand() < 0.5 else idx2

        return population[winner_idx].copy()

    def _environmental_selection(self, population: np.ndarray,
                                 objectives: np.ndarray,
                                 cv: np.ndarray) -> tuple:
        """环境选择（精英保留策略）

        从合并种群中选择 pop_size 个个体作为下一代种群。
        """
        pop_size = self.pop_size
        fitness = self._evaluate_fitness(objectives, cv)

        ranks = fast_non_dominated_sort(fitness)
        max_rank = ranks.max()

        new_indices = []
        current_rank = 0

        while current_rank <= max_rank and len(new_indices) < pop_size:
            front_indices = np.where(ranks == current_rank)[0]

            if len(new_indices) + len(front_indices) <= pop_size:
                new_indices.extend(front_indices.tolist())
            else:
                front_fitness = fitness[front_indices]
                cd = crowding_distance(front_fitness)
                sorted_positions = np.argsort(-cd)
                remaining = pop_size - len(new_indices)
                selected_positions = sorted_positions[:remaining]
                new_indices.extend(front_indices[selected_positions].tolist())

            current_rank += 1

        new_population = population[new_indices]
        new_objectives = objectives[new_indices]
        new_cv = cv[new_indices]

        return new_population, new_objectives, new_cv
