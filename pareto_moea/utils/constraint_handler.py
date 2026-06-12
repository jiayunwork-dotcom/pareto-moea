"""约束处理策略模块"""

import numpy as np


def penalty_function(
    objectives: np.ndarray,
    constraints: np.ndarray,
    penalty_factor: float = 1.0
) -> np.ndarray:
    """罚函数法

    将约束违反量作为惩罚项加到目标函数值上，
    将约束优化问题转化为无约束优化问题。

    Args:
        objectives: 目标函数值，形状为 (n_pop, n_obj)
        constraints: 约束违反量（正值表示违反），形状为 (n_pop, n_constr)
        penalty_factor: 惩罚系数

    Returns:
        惩罚后的目标函数值，形状为 (n_pop, n_obj)
    """
    objectives = np.asarray(objectives, dtype=float)
    constraints = np.asarray(constraints, dtype=float)

    if constraints.size == 0:
        return objectives.copy()

    if constraints.ndim == 1:
        constraint_violation = constraints
    else:
        constraint_violation = np.sum(constraints, axis=1)

    penalty = penalty_factor * constraint_violation

    penalized_objectives = objectives.copy()
    for i in range(objectives.shape[1]):
        penalized_objectives[:, i] += penalty

    return penalized_objectives


def feasibility_rule(
    objectives: np.ndarray,
    constraints: np.ndarray
) -> np.ndarray:
    """可行性规则比较

    基于 Deb 的可行性规则，对个体进行比较排序：
    1. 可行解总是优于不可行解
    2. 两个可行解之间，目标函数值优的解更好
    3. 两个不可行解之间，约束违反量小的解更好

    返回每个个体的排序分数（分数越小越优）。

    Args:
        objectives: 目标函数值，形状为 (n_pop, n_obj)
        constraints: 约束违反量（正值表示违反），形状为 (n_pop, n_constr)

    Returns:
        每个个体的综合排序值，形状为 (n_pop,)
        值越小表示越优
    """
    objectives = np.asarray(objectives, dtype=float)
    constraints = np.asarray(constraints, dtype=float)
    n_pop = objectives.shape[0]

    if constraints.size == 0:
        return np.sum(objectives, axis=1)

    if constraints.ndim == 1:
        cv = constraints
    else:
        cv = np.sum(constraints, axis=1)

    is_feasible = cv <= 1e-10

    score = np.zeros(n_pop)

    max_obj_sum = np.max(np.sum(objectives, axis=1))
    min_obj_sum = np.min(np.sum(objectives, axis=1))
    obj_range = max_obj_sum - min_obj_sum if max_obj_sum > min_obj_sum else 1.0

    max_cv = np.max(cv)
    if max_cv == 0:
        max_cv = 1.0

    for i in range(n_pop):
        if is_feasible[i]:
            score[i] = (np.sum(objectives[i]) - min_obj_sum) / obj_range
        else:
            score[i] = 1.0 + cv[i] / max_cv

    return score


def epsilon_constraint(
    objectives: np.ndarray,
    constraints: np.ndarray,
    epsilon: float = 0.0
) -> np.ndarray:
    """epsilon 约束法

    将约束条件转化为 epsilon 允许的松弛约束，
    只保留约束违反量小于等于 epsilon 的解。
    对于违反约束的解，给予较大的惩罚。

    Args:
        objectives: 目标函数值，形状为 (n_pop, n_obj)
        constraints: 约束违反量（正值表示违反），形状为 (n_pop, n_constr)
        epsilon: 允许的约束违反量阈值

    Returns:
        处理后的目标函数值，形状为 (n_pop, n_obj)
        对于违反 epsilon 的解，目标值设为极大值
    """
    objectives = np.asarray(objectives, dtype=float)
    constraints = np.asarray(constraints, dtype=float)
    n_pop, n_obj = objectives.shape

    if constraints.size == 0:
        return objectives.copy()

    if constraints.ndim == 1:
        cv = constraints
    else:
        cv = np.sum(constraints, axis=1)

    is_epsilon_feasible = cv <= epsilon

    result = objectives.copy()

    for i in range(n_pop):
        if not is_epsilon_feasible[i]:
            penalty = cv[i] - epsilon
            for j in range(n_obj):
                result[i, j] += 1e6 * (1.0 + penalty)

    return result
