"""决策支持模块

提供帕累托前沿的决策分析工具，包括膝点检测、TOPSIS排序和区域筛选。
"""

import numpy as np
from typing import Optional, Tuple, Union


def knee_point_detection(
    objectives: np.ndarray,
    method: str = "angle",
    n_exclude: int = 0,
) -> Tuple[int, np.ndarray]:
    """膝点检测

    在帕累托前沿上找到曲率最大的点（效益最平衡的折中解）。

    Args:
        objectives: 帕累托前沿的目标函数值矩阵，形状为 (n_points, n_obj)
        method: 检测方法，可选 "angle"（基于角度）或 "distance"（基于距离）
        n_exclude: 排除边界点的数量，默认 0（不排除）

    Returns:
        Tuple of (knee_index, knee_objectives)
        - knee_index: 膝点在输入数组中的索引
        - knee_objectives: 膝点的目标函数值

    Raises:
        ValueError: 如果方法不支持或输入数据无效
    """
    objectives = np.asarray(objectives, dtype=float)

    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)

    n_points, n_obj = objectives.shape

    if n_points < 3:
        raise ValueError("帕累托前沿点数过少，至少需要 3 个点才能检测膝点")

    if method == "angle":
        return knee_point_angle_based(objectives, n_exclude)
    elif method == "distance":
        return knee_point_distance_based(objectives, n_exclude)
    else:
        raise ValueError(f"不支持的方法: {method}，可选 'angle' 或 'distance'")


def knee_point_angle_based(
    objectives: np.ndarray,
    n_exclude: int = 0,
) -> Tuple[int, np.ndarray]:
    """基于角度的膝点检测

    通过计算每个点与其前后相邻点形成的夹角，找到夹角最小的点作为膝点。
    适用于二维目标空间。对于高维空间，使用主成分分析降维到二维后再计算。

    Args:
        objectives: 帕累托前沿的目标函数值矩阵，形状为 (n_points, n_obj)
        n_exclude: 排除边界点的数量，默认 0（不排除）

    Returns:
        Tuple of (knee_index, knee_objectives)
    """
    objectives = np.asarray(objectives, dtype=float)
    n_points, n_obj = objectives.shape

    if n_obj == 2:
        sorted_idx = np.argsort(objectives[:, 0])
        sorted_obj = objectives[sorted_idx]
    else:
        sorted_obj = objectives
        sorted_idx = np.arange(n_points)

    angles = np.full(n_points, np.pi)
    start = n_exclude
    end = n_points - n_exclude

    for i in range(start, end):
        if i == 0 or i == n_points - 1:
            continue

        v1 = sorted_obj[i] - sorted_obj[i - 1]
        v2 = sorted_obj[i + 1] - sorted_obj[i]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            continue

        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles[i] = np.arccos(cos_theta)

    knee_idx_local = np.argmin(angles[start:end]) + start
    knee_idx = int(sorted_idx[knee_idx_local])

    return knee_idx, objectives[knee_idx]


def knee_point_distance_based(
    objectives: np.ndarray,
    n_exclude: int = 0,
) -> Tuple[int, np.ndarray]:
    """基于距离的膝点检测

    计算每个点到帕累托前沿两个端点连线的垂直距离，
    距离最大的点即为膝点。

    Args:
        objectives: 帕累托前沿的目标函数值矩阵，形状为 (n_points, n_obj)
        n_exclude: 排除边界点的数量，默认 0（不排除）

    Returns:
        Tuple of (knee_index, knee_objectives)
    """
    objectives = np.asarray(objectives, dtype=float)
    n_points, n_obj = objectives.shape

    if n_obj != 2:
        raise ValueError("基于距离的膝点检测仅支持二维目标空间")

    sorted_idx = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_idx]

    start = n_exclude
    end = n_points - n_exclude

    if end - start < 3:
        raise ValueError("排除边界点后剩余点数过少，至少需要 3 个点")

    p1 = sorted_obj[start]
    p2 = sorted_obj[end - 1]

    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return start, objectives[sorted_idx[start]]

    distances = np.zeros(n_points)

    for i in range(start, end):
        point_vec = sorted_obj[i] - p1
        cross = np.abs(np.cross(line_vec, point_vec))
        distances[i] = cross / line_len

    knee_idx_local = np.argmax(distances[start:end]) + start
    knee_idx = int(sorted_idx[knee_idx_local])

    return knee_idx, objectives[knee_idx]


def topsis(
    objectives: np.ndarray,
    weights: Optional[np.ndarray] = None,
    return_ranks: bool = True,
    minimize: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """TOPSIS 排序

    给定各目标权重后，按离理想点近且离负理想点远的原则排序。

    Args:
        objectives: 目标函数值矩阵，形状为 (n_points, n_obj)
        weights: 各目标的权重向量，形状为 (n_obj,)，默认为等权重
        return_ranks: 是否同时返回排名
        minimize: 是否为最小化问题，默认为 True

    Returns:
        如果 return_ranks 为 True，返回 (scores, ranks)
        - scores: 每个解的 TOPSIS 得分（越大越好），形状为 (n_points,)
        - ranks: 每个解的排名（1 表示最好），形状为 (n_points,)
        否则仅返回 scores
    """
    objectives = np.asarray(objectives, dtype=float)

    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)

    n_points, n_obj = objectives.shape

    if weights is None:
        weights = np.ones(n_obj) / n_obj
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_obj,):
            raise ValueError(f"权重形状应为 ({n_obj},)，实际为 {weights.shape}")
        if np.any(weights < 0):
            raise ValueError("权重不能为负数")
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            raise ValueError("权重之和不能为零")
        weights = weights / weight_sum

    norm_factors = np.sqrt(np.sum(objectives ** 2, axis=0))
    norm_factors[norm_factors == 0] = 1.0
    normalized = objectives / norm_factors

    weighted = normalized * weights

    if minimize:
        ideal = np.min(weighted, axis=0)
        nadir = np.max(weighted, axis=0)
    else:
        ideal = np.max(weighted, axis=0)
        nadir = np.min(weighted, axis=0)

    dist_ideal = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    dist_nadir = np.sqrt(np.sum((weighted - nadir) ** 2, axis=1))

    total_dist = dist_ideal + dist_nadir
    scores = np.where(total_dist == 0, 0.5, dist_nadir / total_dist)

    if return_ranks:
        ranks = n_points - np.argsort(np.argsort(scores))
        return scores, ranks
    else:
        return scores


def region_filter(
    objectives: np.ndarray,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    decision_variables: Optional[np.ndarray] = None,
    return_indices: bool = False,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """区域筛选

    给定目标空间的范围，筛选出指定区域内的解及其决策变量。

    Args:
        objectives: 目标函数值矩阵，形状为 (n_points, n_obj)
        lower_bounds: 目标空间下界，形状为 (n_obj,)，None 表示无下界
        upper_bounds: 目标空间上界，形状为 (n_obj,)，None 表示无上界
        decision_variables: 决策变量矩阵，形状为 (n_points, n_var)，可选
        return_indices: 是否返回筛选后的索引

    Returns:
        根据输入参数返回不同结果：
        - 如果只有 objectives：返回筛选后的目标值矩阵
        - 如果有 decision_variables：返回 (filtered_objectives, filtered_variables)
        - 如果 return_indices=True：额外返回索引数组
    """
    objectives = np.asarray(objectives, dtype=float)

    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)

    n_points, n_obj = objectives.shape

    mask = np.ones(n_points, dtype=bool)

    if lower_bounds is not None:
        lower_bounds = np.asarray(lower_bounds, dtype=float)
        if lower_bounds.shape != (n_obj,):
            raise ValueError(
                f"下界形状应为 ({n_obj},)，实际为 {lower_bounds.shape}"
            )
        mask &= np.all(objectives >= lower_bounds, axis=1)

    if upper_bounds is not None:
        upper_bounds = np.asarray(upper_bounds, dtype=float)
        if upper_bounds.shape != (n_obj,):
            raise ValueError(
                f"上界形状应为 ({n_obj},)，实际为 {upper_bounds.shape}"
            )
        mask &= np.all(objectives <= upper_bounds, axis=1)

    indices = np.where(mask)[0]
    filtered_obj = objectives[indices]

    if decision_variables is not None:
        decision_variables = np.asarray(decision_variables)
        if decision_variables.shape[0] != n_points:
            raise ValueError(
                f"决策变量数量 ({decision_variables.shape[0]}) 与目标值数量 ({n_points}) 不匹配"
            )
        filtered_var = decision_variables[indices]

        if return_indices:
            return filtered_obj, filtered_var, indices
        else:
            return filtered_obj, filtered_var
    else:
        if return_indices:
            return filtered_obj, indices
        else:
            return filtered_obj
