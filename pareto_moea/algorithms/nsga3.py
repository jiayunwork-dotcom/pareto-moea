"""NSGA-III 算法实现"""

import numpy as np
from typing import Optional
from .base import Algorithm


class NSGA3(Algorithm):
    """基于参考点的非支配排序遗传算法 III (NSGA-III)

    NSGA-III 使用基于参考点的选择机制替代 NSGA-II 的拥挤距离，
    更适合处理高维多目标优化问题。

    核心特性:
        - Das-Dennis 参考点生成
        - 自适应目标归一化
        - 关联操作 (Association)
        - 小生境保护 (Niching)

    Args:
        pop_size: 种群大小
        n_gen: 进化代数
        n_divisions: 参考点划分数 (Das-Dennis 方法的参数)
        crossover_prob: 交叉概率
        crossover_eta: SBX 交叉分布指数
        mutation_prob: 变异概率 (默认为 1/n_var)
        mutation_eta: 多项式变异分布指数
        constraint_strategy: 约束处理策略
        penalty_factor: 惩罚因子
        seed: 随机种子
    """

    def __init__(
        self,
        pop_size: int = 100,
        n_gen: int = 100,
        n_divisions: int = 12,
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
        self.n_divisions = n_divisions
        self.reference_points = None

    def setup(self, problem):
        """设置优化问题并生成参考点"""
        super().setup(problem)
        from ..utils.pareto_utils import uniform_reference_points

        self.reference_points = uniform_reference_points(
            problem.n_obj, self.n_divisions
        )

        n_ref_points = len(self.reference_points)
        if self.pop_size < n_ref_points:
            self.pop_size = n_ref_points

    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        return self._initialize_random()

    def _evolve(self, population: np.ndarray, objectives: np.ndarray,
                cv: np.ndarray, gen: int) -> tuple:
        """执行一代进化

        Args:
            population: 当前种群，形状为 (pop_size, n_var)
            objectives: 目标函数值，形状为 (pop_size, n_obj)
            cv: 约束违反量，形状为 (pop_size,)
            gen: 当前代数

        Returns:
            (new_population, new_objectives, new_cv)
        """
        offspring = self._create_offspring(population, objectives, cv)
        offspring_obj = self.problem.evaluate(offspring)
        offspring_cv = self.problem.constraint_violation(offspring)

        merged_pop = np.vstack([population, offspring])
        merged_obj = np.vstack([objectives, offspring_obj])
        merged_cv = np.concatenate([cv, offspring_cv])

        fitness = self._evaluate_fitness(merged_obj, merged_cv)
        selected = self._selection(merged_pop, fitness, merged_obj)

        new_population = merged_pop[selected]
        new_objectives = merged_obj[selected]
        new_cv = merged_cv[selected]

        return new_population, new_objectives, new_cv

    def _create_offspring(self, population: np.ndarray,
                          objectives: np.ndarray,
                          cv: np.ndarray) -> np.ndarray:
        """创建子代种群"""
        from ..utils.operators import binary_tournament_selection

        n_pop = len(population)
        fitness = self._evaluate_fitness(objectives, cv)
        agg_fitness = np.sum(fitness, axis=1) if fitness.ndim > 1 else fitness

        offspring = np.zeros_like(population)

        for i in range(0, n_pop, 2):
            parent1 = binary_tournament_selection(
                population, agg_fitness, tournament_size=2
            )
            parent2 = binary_tournament_selection(
                population, agg_fitness, tournament_size=2
            )

            child1, child2 = self._sbx_crossover(parent1, parent2)
            child1 = self._polynomial_mutation(child1)
            child2 = self._polynomial_mutation(child2)

            offspring[i] = child1
            if i + 1 < n_pop:
                offspring[i + 1] = child2

        return offspring

    def _selection(self, population: np.ndarray, fitness: np.ndarray,
                   objectives: np.ndarray) -> np.ndarray:
        """NSGA-III 选择机制

        Args:
            population: 合并后的种群
            fitness: 适应度值 (考虑约束)
            objectives: 目标函数值

        Returns:
            选中个体的索引
        """
        from ..utils.pareto_utils import fast_non_dominated_sort

        n_select = self.pop_size
        n_total = len(population)

        ranks = fast_non_dominated_sort(fitness)

        fronts = []
        max_rank = int(np.max(ranks))
        for r in range(max_rank + 1):
            front_indices = np.where(ranks == r)[0]
            fronts.append(front_indices)

        selected = []
        current_size = 0
        last_front_idx = 0

        for i, front in enumerate(fronts):
            if current_size + len(front) <= n_select:
                selected.extend(front.tolist())
                current_size += len(front)
            else:
                last_front_idx = i
                break

        if current_size == n_select:
            return np.array(selected)

        last_front = fronts[last_front_idx]
        n_remaining = n_select - current_size

        chosen = self._niching_selection(
            objectives, np.array(selected), last_front, n_remaining
        )

        selected.extend(chosen.tolist())

        return np.array(selected)

    def _niching_selection(self, objectives: np.ndarray,
                           selected_indices: np.ndarray,
                           last_front: np.ndarray,
                           n_remaining: int) -> np.ndarray:
        """基于小生境的选择

        对最后一个前沿层使用参考点小生境机制进行选择。

        Args:
            objectives: 所有个体的目标值
            selected_indices: 已选中的个体索引
            last_front: 最后一个前沿层的个体索引
            n_remaining: 需要从最后一个前沿层选择的个体数

        Returns:
            从最后一个前沿层选中的个体索引
        """
        n_obj = self.problem.n_obj
        all_indices = np.concatenate([selected_indices, last_front])
        all_objectives = objectives[all_indices]

        ideal_point = np.min(all_objectives, axis=0)

        normalized_obj = self._normalize_objectives(
            all_objectives, ideal_point
        )

        associations, distances = self._associate_to_reference_points(
            normalized_obj
        )

        n_selected = len(selected_indices)
        niche_count = np.zeros(len(self.reference_points))

        for i in range(n_selected):
            ref_idx = associations[i]
            if ref_idx >= 0:
                niche_count[ref_idx] += 1

        last_front_assoc = associations[n_selected:]
        last_front_dist = distances[n_selected:]

        chosen = []
        available = np.ones(len(last_front), dtype=bool)

        for _ in range(n_remaining):
            min_count = np.min(niche_count)
            candidate_refs = np.where(niche_count == min_count)[0]

            best_ref = -1
            best_idx = -1
            best_dist = np.inf

            for ref_idx in candidate_refs:
                candidates_in_ref = np.where(
                    (last_front_assoc == ref_idx) & available
                )[0]

                if len(candidates_in_ref) > 0:
                    closest = candidates_in_ref[
                        np.argmin(last_front_dist[candidates_in_ref])
                    ]
                    if last_front_dist[closest] < best_dist:
                        best_dist = last_front_dist[closest]
                        best_idx = closest
                        best_ref = ref_idx

            if best_idx >= 0:
                chosen.append(last_front[best_idx])
                available[best_idx] = False
                niche_count[best_ref] += 1
            else:
                remaining = np.where(available)[0]
                if len(remaining) > 0:
                    pick_idx = self.rng.choice(remaining)
                    chosen.append(last_front[pick_idx])
                    available[pick_idx] = False
                    ref_idx = last_front_assoc[pick_idx]
                    if ref_idx >= 0:
                        niche_count[ref_idx] += 1
                else:
                    break

        return np.array(chosen)

    def _normalize_objectives(self, objectives: np.ndarray,
                              ideal_point: np.ndarray) -> np.ndarray:
        """自适应目标归一化

        使用极值点和超平面对目标进行归一化。

        Args:
            objectives: 目标函数值，形状为 (n_pop, n_obj)
            ideal_point: 理想点 (每个目标的最小值)

        Returns:
            归一化后的目标值
        """
        n_obj = objectives.shape[1]

        translated = objectives - ideal_point

        extreme_points = self._find_extreme_points(translated)

        intercepts = np.zeros(n_obj)
        try:
            hyperplane = np.linalg.solve(extreme_points, np.ones(n_obj))
            intercepts = 1.0 / hyperplane
        except np.linalg.LinAlgError:
            intercepts = np.max(translated, axis=0)

        intercepts = np.where(intercepts <= 0, 1e-10, intercepts)

        normalized = translated / intercepts

        return normalized

    def _find_extreme_points(self, translated_obj: np.ndarray) -> np.ndarray:
        """寻找极值点

        使用 Achievement Scalarizing Function (ASF) 寻找极值点。

        Args:
            translated_obj: 平移后的目标值 (已减去理想点)

        Returns:
            极值点矩阵，形状为 (n_obj, n_obj)
        """
        n_obj = translated_obj.shape[1]
        extreme_points = np.zeros((n_obj, n_obj))

        epsilon = 1e-6

        for i in range(n_obj):
            weights = np.full(n_obj, epsilon)
            weights[i] = 1.0

            asf_values = np.max(translated_obj / weights, axis=1)
            best_idx = np.argmin(asf_values)
            extreme_points[i] = translated_obj[best_idx]

        return extreme_points

    def _associate_to_reference_points(self, normalized_obj: np.ndarray
                                        ) -> tuple:
        """关联操作：将每个个体关联到最近的参考点

        Args:
            normalized_obj: 归一化后的目标值

        Returns:
            (associations, distances)
            - associations: 每个个体关联的参考点索引
            - distances: 每个个体到对应参考点的垂直距离
        """
        n_pop = normalized_obj.shape[0]
        n_ref = len(self.reference_points)

        associations = np.full(n_pop, -1, dtype=int)
        distances = np.full(n_pop, np.inf)

        for i in range(n_pop):
            point = normalized_obj[i]
            min_dist = np.inf
            best_ref = -1

            for j in range(n_ref):
                ref = self.reference_points[j]
                ref_norm = np.linalg.norm(ref)
                if ref_norm < 1e-10:
                    continue

                projection = np.dot(point, ref) / (ref_norm ** 2)
                projected_point = projection * ref
                dist = np.linalg.norm(point - projected_point)

                if dist < min_dist:
                    min_dist = dist
                    best_ref = j

            associations[i] = best_ref
            distances[i] = min_dist

        return associations, distances

    def get_params(self) -> dict:
        """获取算法参数"""
        params = super().get_params()
        params['n_divisions'] = self.n_divisions
        return params
