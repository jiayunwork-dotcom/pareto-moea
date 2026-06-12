"""SPEA2算法实现"""

import numpy as np
from .base import Algorithm
from ..utils.operators import sbx_crossover, polynomial_mutation


class SPEA2(Algorithm):
    """强度 Pareto 进化算法 2 (Strength Pareto Evolutionary Algorithm 2)

    核心组件：
    - 强度值计算 (Strength Value)
    - 原始适应度 (Raw Fitness)
    - k近邻密度估计 (k-Nearest Neighbor Density Estimation)
    - 精细适应度 (Fitness = Raw Fitness + Density)
    - 归档集截断策略 (Archive Truncation)
    - 二元锦标赛选择 (Binary Tournament Selection)
    - SBX交叉 (Simulated Binary Crossover)
    - 多项式变异 (Polynomial Mutation)
    """

    def __init__(
        self,
        pop_size: int = 100,
        n_gen: int = 100,
        archive_size: int = None,
        k_neighbors: int = None,
        crossover_prob: float = 0.9,
        crossover_eta: float = 20.0,
        mutation_prob: float = None,
        mutation_eta: float = 20.0,
        constraint_strategy: str = "feasibility_rule",
        penalty_factor: float = 100.0,
        seed: int = None
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
        self.archive_size = archive_size if archive_size is not None else pop_size
        self._k_neighbors = k_neighbors
        self.archive_pop = None
        self.archive_obj = None
        self.archive_cv = None

    def setup(self, problem):
        """设置优化问题"""
        super().setup(problem)
        if self._k_neighbors is None:
            total_size = self.pop_size + self.archive_size
            self.k_neighbors = int(np.sqrt(total_size))
        else:
            self.k_neighbors = self._k_neighbors

    def get_params(self) -> dict:
        """获取算法参数"""
        params = super().get_params()
        params.update({
            'archive_size': self.archive_size,
            'k_neighbors': self.k_neighbors
        })
        return params

    def _initialize_population(self) -> np.ndarray:
        """初始化种群和归档集"""
        pop = self._initialize_random()
        self.archive_pop = pop.copy()
        self.archive_obj = self.problem.evaluate(pop)
        self.archive_cv = self.problem.constraint_violation(pop)
        return pop

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
        merged_pop = np.vstack([population, self.archive_pop])
        merged_obj = np.vstack([objectives, self.archive_obj])
        merged_cv = np.concatenate([cv, self.archive_cv])

        fitness = self._evaluate_fitness(merged_obj, merged_cv)
        strength = self._compute_strength(fitness)
        raw_fitness = self._compute_raw_fitness(fitness, strength)
        density = self._compute_density(fitness)
        fine_fitness = raw_fitness + density

        self.archive_pop, self.archive_obj, self.archive_cv = (
            self._environmental_selection(
                merged_pop, merged_obj, merged_cv, fine_fitness, raw_fitness
            )
        )

        archive_fitness = self._evaluate_fitness(self.archive_obj, self.archive_cv)
        archive_strength = self._compute_strength(archive_fitness)
        archive_raw = self._compute_raw_fitness(archive_fitness, archive_strength)
        archive_density = self._compute_density(archive_fitness)
        archive_fine_fitness = archive_raw + archive_density

        offspring = self._create_offspring(
            self.archive_pop, archive_fine_fitness
        )
        offspring_obj = self.problem.evaluate(offspring)
        offspring_cv = self.problem.constraint_violation(offspring)

        return offspring, offspring_obj, offspring_cv

    def _compute_strength(self, fitness: np.ndarray) -> np.ndarray:
        """计算强度值

        强度值 S(i) 表示个体 i 支配的个体数量。

        Args:
            fitness: 适应度值矩阵，形状为 (n_pop, n_obj)

        Returns:
            每个个体的强度值，形状为 (n_pop,)
        """
        n_pop = len(fitness)
        strength = np.zeros(n_pop, dtype=int)

        for i in range(n_pop):
            for j in range(n_pop):
                if i != j and self._dominates(fitness[i], fitness[j]):
                    strength[i] += 1

        return strength

    def _compute_raw_fitness(self, fitness: np.ndarray,
                             strength: np.ndarray) -> np.ndarray:
        """计算原始适应度

        原始适应度 R(i) 等于所有支配个体 i 的个体的强度值之和。
        R(i) 越小越好，R(i) = 0 表示非支配解。

        Args:
            fitness: 适应度值矩阵，形状为 (n_pop, n_obj)
            strength: 强度值数组，形状为 (n_pop,)

        Returns:
            每个个体的原始适应度，形状为 (n_pop,)
        """
        n_pop = len(fitness)
        raw_fitness = np.zeros(n_pop, dtype=float)

        for i in range(n_pop):
            for j in range(n_pop):
                if i != j and self._dominates(fitness[j], fitness[i]):
                    raw_fitness[i] += strength[j]

        return raw_fitness

    def _compute_density(self, fitness: np.ndarray) -> np.ndarray:
        """计算k近邻密度估计值

        密度估计值 D(i) = 1 / (sigma_i^k + 2)
        其中 sigma_i^k 是个体 i 到第 k 个最近邻的距离。

        Args:
            fitness: 适应度值矩阵，形状为 (n_pop, n_obj)

        Returns:
            每个个体的密度估计值，形状为 (n_pop,)
        """
        n_pop = len(fitness)
        k = min(self.k_neighbors, n_pop - 1)

        distances = np.zeros((n_pop, n_pop))
        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                dist = np.sqrt(np.sum((fitness[i] - fitness[j]) ** 2))
                distances[i, j] = dist
                distances[j, i] = dist

        density = np.zeros(n_pop)
        for i in range(n_pop):
            sorted_dists = np.sort(distances[i])
            kth_dist = sorted_dists[k]
            density[i] = 1.0 / (kth_dist + 2.0)

        return density

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """判断解 a 是否支配解 b（最小化问题）"""
        return np.all(a <= b) and np.any(a < b)

    def _environmental_selection(self, population: np.ndarray,
                                 objectives: np.ndarray,
                                 cv: np.ndarray,
                                 fine_fitness: np.ndarray,
                                 raw_fitness: np.ndarray) -> tuple:
        """环境选择：从合并种群中选择归档集

        步骤：
        1. 选择所有非支配解（raw_fitness < 1）
        2. 如果非支配解数量等于 archive_size，直接返回
        3. 如果非支配解数量少于 archive_size，按精细适应度排序补充
        4. 如果非支配解数量超过 archive_size，执行截断策略

        Args:
            population: 合并种群，形状为 (n_total, n_var)
            objectives: 目标函数值，形状为 (n_total, n_obj)
            cv: 约束违反值，形状为 (n_total, n_constr)
            fine_fitness: 精细适应度，形状为 (n_total,)
            raw_fitness: 原始适应度，形状为 (n_total,)

        Returns:
            (archive_pop, archive_obj, archive_cv)
        """
        n_total = len(population)
        archive_size = self.archive_size

        nondominated_mask = raw_fitness < 1.0
        nondominated_indices = np.where(nondominated_mask)[0]
        n_nondominated = len(nondominated_indices)

        if n_nondominated == archive_size:
            selected = nondominated_indices
        elif n_nondominated < archive_size:
            sorted_indices = np.argsort(fine_fitness)
            selected = []
            for idx in sorted_indices:
                if len(selected) >= archive_size:
                    break
                if idx not in selected:
                    selected.append(idx)
            selected = np.array(selected)
        else:
            selected = self._archive_truncation(
                objectives[nondominated_indices], archive_size
            )
            selected = nondominated_indices[selected]

        return (
            population[selected].copy(),
            objectives[selected].copy(),
            cv[selected].copy()
        )

    def _archive_truncation(self, objectives: np.ndarray,
                            target_size: int) -> np.ndarray:
        """归档集截断策略

        当非支配解数量超过 archive_size 时，逐一移除密度贡献最小的个体。
        每次移除后重新计算受影响个体的距离。

        Args:
            objectives: 非支配解的目标值，形状为 (n_nondominated, n_obj)
            target_size: 目标归档集大小

        Returns:
            选中的索引数组，形状为 (target_size,)
        """
        n = len(objectives)
        if n <= target_size:
            return np.arange(n)

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((objectives[i] - objectives[j]) ** 2))
                distances[i, j] = dist
                distances[j, i] = dist

        remaining = list(range(n))
        k = self.k_neighbors

        while len(remaining) > target_size:
            m = len(remaining)
            current_k = min(k, m - 1)

            kth_distances = np.zeros(m)
            for i_idx in range(m):
                i = remaining[i_idx]
                dists = [distances[i][remaining[j_idx]]
                         for j_idx in range(m) if j_idx != i_idx]
                dists.sort()
                kth_distances[i_idx] = dists[current_k - 1]

            min_dist_idx = np.argmin(kth_distances)
            del remaining[min_dist_idx]

        return np.array(remaining)

    def _create_offspring(self, population: np.ndarray,
                          fitness: np.ndarray) -> np.ndarray:
        """创建子代种群

        使用基于精细适应度的二元锦标赛选择，然后进行SBX交叉和多项式变异。

        Args:
            population: 父代种群，形状为 (n_pop, n_var)
            fitness: 精细适应度值，形状为 (n_pop,)，越小越好

        Returns:
            子代种群，形状为 (pop_size, n_var)
        """
        n_parents = len(population)
        offspring = np.zeros((self.pop_size, self.problem.n_var))

        for i in range(0, self.pop_size, 2):
            parent1 = self._binary_tournament(population, fitness)
            parent2 = self._binary_tournament(population, fitness)

            child1, child2 = sbx_crossover(
                parent1, parent2,
                self.problem.xl, self.problem.xu,
                self.crossover_eta, self.crossover_prob,
                self.rng
            )

            child1 = polynomial_mutation(
                child1,
                self.problem.xl, self.problem.xu,
                self.mutation_eta, self.mutation_prob,
                self.rng
            )
            child2 = polynomial_mutation(
                child2,
                self.problem.xl, self.problem.xu,
                self.mutation_eta, self.mutation_prob,
                self.rng
            )

            offspring[i] = child1
            if i + 1 < self.pop_size:
                offspring[i + 1] = child2

        return offspring

    def _binary_tournament(self, population: np.ndarray,
                           fitness: np.ndarray) -> np.ndarray:
        """基于精细适应度的二元锦标赛选择

        精细适应度越小越好。

        Args:
            population: 种群，形状为 (n_pop, n_var)
            fitness: 精细适应度值，形状为 (n_pop,)

        Returns:
            选中的个体
        """
        n_pop = len(population)
        indices = self.rng.choice(n_pop, size=2, replace=False)

        idx1, idx2 = indices[0], indices[1]

        if fitness[idx1] < fitness[idx2]:
            winner_idx = idx1
        elif fitness[idx1] > fitness[idx2]:
            winner_idx = idx2
        else:
            winner_idx = idx1 if self.rng.rand() < 0.5 else idx2

        return population[winner_idx].copy()
