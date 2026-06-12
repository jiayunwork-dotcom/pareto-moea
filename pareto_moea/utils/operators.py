"""遗传算子模块"""

import numpy as np


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta: float = 20.0,
    prob: float = 1.0,
    rng: np.random.RandomState = None
) -> tuple:
    """模拟二进制交叉 (Simulated Binary Crossover, SBX)

    对两个父代个体执行 SBX 交叉操作，产生两个子代个体。

    Args:
        parent1: 第一个父代个体，形状为 (n_var,)
        parent2: 第二个父代个体，形状为 (n_var,)
        xl: 变量下界，形状为 (n_var,)
        xu: 变量上界，形状为 (n_var,)
        eta: 分布指数，值越大子代越接近父代
        prob: 交叉概率
        rng: 随机数生成器，为None时使用全局np.random

    Returns:
        (child1, child2) 两个子代个体
    """
    if rng is None:
        rng = np.random

    parent1 = np.asarray(parent1, dtype=float)
    parent2 = np.asarray(parent2, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    n_var = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    if rng.rand() > prob:
        return child1, child2

    for i in range(n_var):
        if rng.rand() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                y1 = min(parent1[i], parent2[i])
                y2 = max(parent1[i], parent2[i])

                yl = xl[i]
                yu = xu[i]

                rand = rng.rand()

                beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                alpha = 2.0 - beta ** (-(eta + 1.0))
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                alpha = 2.0 - beta ** (-(eta + 1.0))
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                c1 = np.clip(c1, yl, yu)
                c2 = np.clip(c2, yl, yu)

                if rng.rand() <= 0.5:
                    child1[i] = c2
                    child2[i] = c1
                else:
                    child1[i] = c1
                    child2[i] = c2

    return child1, child2


def polynomial_mutation(
    individual: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta: float = 20.0,
    prob: float = None,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """多项式变异 (Polynomial Mutation)

    对个体执行多项式变异操作。

    Args:
        individual: 个体，形状为 (n_var,)
        xl: 变量下界，形状为 (n_var,)
        xu: 变量上界，形状为 (n_var,)
        eta: 分布指数，值越大变异幅度越小
        prob: 每个变量的变异概率，默认为 1/n_var
        rng: 随机数生成器，为None时使用全局np.random

    Returns:
        变异后的个体
    """
    if rng is None:
        rng = np.random

    individual = np.asarray(individual, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    n_var = len(individual)
    if prob is None:
        prob = 1.0 / n_var

    mutant = individual.copy()

    for i in range(n_var):
        if rng.rand() <= prob:
            y = individual[i]
            yl = xl[i]
            yu = xu[i]

            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)

            rand = rng.rand()
            mut_pow = 1.0 / (eta + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val ** mut_pow

            y = y + deltaq * (yu - yl)
            y = np.clip(y, yl, yu)
            mutant[i] = y

    return mutant


def binary_tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int = 2,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """二元锦标赛选择

    从种群中通过锦标赛选择选出一个个体。
    适用于单目标优化的适应度选择。

    Args:
        population: 种群，形状为 (n_pop, n_var)
        fitness: 适应度值，形状为 (n_pop,)，越小越好（最小化）
        tournament_size: 锦标赛规模
        rng: 随机数生成器，为None时使用全局np.random

    Returns:
        选中的个体
    """
    if rng is None:
        rng = np.random

    population = np.asarray(population, dtype=float)
    fitness = np.asarray(fitness, dtype=float)

    n_pop = len(population)
    indices = rng.choice(n_pop, size=tournament_size, replace=False)

    best_idx = indices[0]
    for idx in indices[1:]:
        if fitness[idx] < fitness[best_idx]:
            best_idx = idx

    return population[best_idx].copy()
