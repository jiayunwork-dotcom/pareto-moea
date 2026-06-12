"""算法基类"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class AlgorithmResult:
    """算法运行结果"""

    problem_name: str
    algorithm_name: str
    pop_size: int
    n_gen: int
    final_population: np.ndarray
    final_objectives: np.ndarray
    final_constraint_violation: np.ndarray
    pareto_front: np.ndarray
    pareto_set: np.ndarray
    history: Dict[str, List] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    runtime: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Algorithm(ABC):
    """多目标进化算法基类"""

    def __init__(
        self,
        pop_size: int = 100,
        n_gen: int = 100,
        crossover_prob: float = 0.9,
        crossover_eta: float = 20.0,
        mutation_prob: Optional[float] = None,
        mutation_eta: float = 20.0,
        constraint_strategy: str = "feasibility_rule",
        penalty_factor: float = 100.0,
        seed: Optional[int] = None
    ):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob
        self.mutation_eta = mutation_eta
        self.constraint_strategy = constraint_strategy
        self.penalty_factor = penalty_factor
        self.seed = seed

        self.problem = None
        self.rng = np.random.RandomState(seed)

        self._paused = False
        self._stopped = False
        self._current_gen = 0
        self._callback = None

    def setup(self, problem):
        """设置优化问题"""
        self.problem = problem
        if self.mutation_prob is None:
            self.mutation_prob = 1.0 / problem.n_var

    def pause(self):
        """暂停算法"""
        self._paused = True

    def resume(self):
        """继续算法"""
        self._paused = False

    def stop(self):
        """终止算法"""
        self._stopped = True

    def is_paused(self):
        return self._paused

    def is_stopped(self):
        return self._stopped

    def current_generation(self):
        return self._current_gen

    def set_callback(self, callback: Callable):
        """设置每代回调函数

        callback(algorithm, gen, population, objectives, constraint_violation)
        """
        self._callback = callback

    @abstractmethod
    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        pass

    @abstractmethod
    def _evolve(self, population: np.ndarray, objectives: np.ndarray,
                cv: np.ndarray, gen: int) -> tuple:
        """执行一代进化

        Returns:
            (new_population, new_objectives, new_cv)
        """
        pass

    def run(self, problem, verbose: bool = False) -> AlgorithmResult:
        """运行算法

        Args:
            problem: 优化问题
            verbose: 是否打印进度

        Returns:
            AlgorithmResult 对象
        """
        self.setup(problem)
        start_time = time.time()

        self._current_gen = 0
        self._stopped = False
        self._paused = False

        population = self._initialize_population()
        objectives = problem.evaluate(population)
        cv = problem.constraint_violation(population)

        history = {
            'populations': [population.copy()],
            'objectives': [objectives.copy()],
            'cv': [cv.copy()],
            'metrics': []
        }

        if self._callback:
            self._callback(self, 0, population, objectives, cv)

        for gen in range(1, self.n_gen + 1):
            while self._paused and not self._stopped:
                time.sleep(0.1)

            if self._stopped:
                break

            self._current_gen = gen

            population, objectives, cv = self._evolve(
                population, objectives, cv, gen
            )

            history['populations'].append(population.copy())
            history['objectives'].append(objectives.copy())
            history['cv'].append(cv.copy())

            if self._callback:
                self._callback(self, gen, population, objectives, cv)

            if verbose and gen % max(1, self.n_gen // 10) == 0:
                print(f"Gen {gen}/{self.n_gen} completed")

        runtime = time.time() - start_time

        from ..utils.pareto_utils import pareto_front
        feasible_mask = cv <= 1e-10
        if np.any(feasible_mask):
            feasible_obj = objectives[feasible_mask]
            feasible_pop = population[feasible_mask]
            pf_indices = pareto_front(feasible_obj, return_indices=True)
            final_pf = feasible_obj[pf_indices]
            final_ps = feasible_pop[pf_indices]
        else:
            pf_indices = pareto_front(objectives, return_indices=True)
            final_pf = objectives[pf_indices]
            final_ps = population[pf_indices]

        result = AlgorithmResult(
            problem_name=problem.name,
            algorithm_name=self.__class__.__name__,
            pop_size=self.pop_size,
            n_gen=self._current_gen,
            final_population=population,
            final_objectives=objectives,
            final_constraint_violation=cv,
            pareto_front=final_pf,
            pareto_set=final_ps,
            history=history,
            runtime=runtime,
            params=self.get_params(),
            timestamp=time.time()
        )

        return result

    def get_params(self) -> Dict[str, Any]:
        """获取算法参数"""
        return {
            'pop_size': self.pop_size,
            'n_gen': self.n_gen,
            'crossover_prob': self.crossover_prob,
            'crossover_eta': self.crossover_eta,
            'mutation_prob': self.mutation_prob,
            'mutation_eta': self.mutation_eta,
            'constraint_strategy': self.constraint_strategy,
            'penalty_factor': self.penalty_factor,
            'seed': self.seed
        }

    def _evaluate_fitness(self, objectives: np.ndarray,
                          cv: np.ndarray) -> np.ndarray:
        """根据约束处理策略评估适应度（用于选择）

        返回值越小越好。
        """
        from ..utils.constraint_handler import (
            penalty_function, feasibility_rule, epsilon_constraint
        )

        if self.problem.n_constr == 0 or np.all(cv <= 1e-10):
            return objectives

        if self.constraint_strategy == "penalty":
            return penalty_function(objectives, cv, self.penalty_factor)
        elif self.constraint_strategy == "feasibility_rule":
            return feasibility_rule(objectives, cv)
        elif self.constraint_strategy == "epsilon":
            epsilon = max(0.01, 0.1 * (1 - self._current_gen / self.n_gen))
            return epsilon_constraint(objectives, cv, epsilon)
        else:
            return objectives

    def _initialize_random(self) -> np.ndarray:
        """随机初始化种群"""
        pop = np.zeros((self.pop_size, self.problem.n_var))
        for i in range(self.problem.n_var):
            pop[:, i] = self.rng.uniform(
                self.problem.xl[i], self.problem.xu[i], self.pop_size
            )
        return pop

    def _sbx_crossover(self, parent1: np.ndarray,
                       parent2: np.ndarray) -> tuple:
        """SBX交叉"""
        from ..utils.operators import sbx_crossover
        return sbx_crossover(
            parent1, parent2,
            self.problem.xl, self.problem.xu,
            self.crossover_eta, self.crossover_prob,
            self.rng
        )

    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """多项式变异"""
        from ..utils.operators import polynomial_mutation
        return polynomial_mutation(
            individual,
            self.problem.xl, self.problem.xu,
            self.mutation_eta, self.mutation_prob,
            self.rng
        )
