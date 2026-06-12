"""帕累托前沿计算工具函数"""

import numpy as np
from itertools import combinations


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """判断解 a 是否支配解 b

    最小化问题：a 支配 b 当且仅当 a 在所有目标上都不劣于 b，
    且至少在一个目标上严格优于 b。

    Args:
        a: 解 a 的目标函数值，形状为 (n_obj,)
        b: 解 b 的目标函数值，形状为 (n_obj,)

    Returns:
        True 如果 a 支配 b，否则 False
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sort(objectives: np.ndarray) -> list:
    """非支配排序（基础版本）

    将种群中的个体按 Pareto 支配关系分层，
    返回每个前沿层的索引列表。

    Args:
        objectives: 目标函数值矩阵，形状为 (n_pop, n_obj)

    Returns:
        列表，每个元素是一个前沿层的索引数组，
        按 Pareto 等级从高到低排列
    """
    objectives = np.asarray(objectives, dtype=float)
    n_pop = objectives.shape[0]

    dominated_count = np.zeros(n_pop, dtype=int)
    domination_set = [[] for _ in range(n_pop)]

    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            if dominates(objectives[i], objectives[j]):
                dominated_count[j] += 1
                domination_set[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                dominated_count[i] += 1
                domination_set[j].append(i)

    fronts = []
    current_front = np.where(dominated_count == 0)[0].tolist()

    while current_front:
        fronts.append(np.array(current_front))
        next_front = []
        for i in current_front:
            for j in domination_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def fast_non_dominated_sort(objectives: np.ndarray) -> np.ndarray:
    """快速非支配排序

    返回每个个体的 Pareto 等级（从 0 开始）。

    Args:
        objectives: 目标函数值矩阵，形状为 (n_pop, n_obj)

    Returns:
        每个个体的 Pareto 等级，形状为 (n_pop,)
    """
    objectives = np.asarray(objectives, dtype=float)
    n_pop = objectives.shape[0]

    ranks = np.zeros(n_pop, dtype=int)
    dominated_count = np.zeros(n_pop, dtype=int)
    domination_set = [[] for _ in range(n_pop)]

    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            if dominates(objectives[i], objectives[j]):
                dominated_count[j] += 1
                domination_set[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                dominated_count[i] += 1
                domination_set[j].append(i)

    current_rank = 0
    current_front = np.where(dominated_count == 0)[0].tolist()

    while current_front:
        for i in current_front:
            ranks[i] = current_rank
        next_front = []
        for i in current_front:
            for j in domination_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        current_rank += 1

    return ranks


def pareto_front(objectives: np.ndarray, return_indices: bool = False):
    """提取帕累托前沿

    从种群中提取非支配解（第一前沿）。

    Args:
        objectives: 目标函数值矩阵，形状为 (n_pop, n_obj)
        return_indices: 是否返回索引

    Returns:
        如果 return_indices 为 True，返回帕累托前沿的索引数组；
        否则返回帕累托前沿的目标值矩阵
    """
    objectives = np.asarray(objectives, dtype=float)
    fronts = non_dominated_sort(objectives)

    if return_indices:
        return fronts[0]
    else:
        return objectives[fronts[0]]


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """计算拥挤距离

    计算每个个体在其所在前沿中的拥挤距离，
    用于保持种群的多样性。

    Args:
        objectives: 同一前沿内个体的目标函数值矩阵，
                    形状为 (n_pop, n_obj)

    Returns:
        每个个体的拥挤距离，形状为 (n_pop,)
    """
    objectives = np.asarray(objectives, dtype=float)
    n_pop, n_obj = objectives.shape

    if n_pop <= 2:
        return np.full(n_pop, np.inf)

    distances = np.zeros(n_pop)

    for m in range(n_obj):
        sorted_indices = np.argsort(objectives[:, m])
        sorted_obj = objectives[sorted_indices, m]

        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        obj_range = sorted_obj[-1] - sorted_obj[0]
        if obj_range > 0:
            for i in range(1, n_pop - 1):
                distances[sorted_indices[i]] += (
                    sorted_obj[i + 1] - sorted_obj[i - 1]
                ) / obj_range

    return distances


def uniform_reference_points(n_obj: int, n_divisions: int) -> np.ndarray:
    """Das-Dennis 参考点生成

    在目标空间的单纯形上生成均匀分布的参考点。
    常用于 NSGA-III、MOEA/D 等算法。

    Args:
        n_obj: 目标个数
        n_divisions: 每个目标方向上的等分数

    Returns:
        参考点矩阵，形状为 (n_points, n_obj)，
        每个参考点的各分量之和为 1
    """
    if n_obj == 1:
        return np.ones((1, 1))

    points = []
    h = n_divisions

    def _generate_recursive(obj_idx: int, remaining: int, current: list):
        if obj_idx == n_obj - 1:
            current.append(remaining / h)
            points.append(current.copy())
            current.pop()
            return

        for val in range(remaining + 1):
            current.append(val / h)
            _generate_recursive(obj_idx + 1, remaining - val, current)
            current.pop()

    _generate_recursive(0, h, [])

    return np.array(points)
