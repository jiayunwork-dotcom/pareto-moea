"""多目标优化性能指标"""

import numpy as np
from typing import Optional


def generational_distance(approx_front: np.ndarray, true_front: np.ndarray, p: float = 2.0) -> float:
    """计算世代距离 (Generational Distance, GD)

    GD 衡量近似前沿到真实前沿的平均距离，反映收敛性。
    值越小越好，GD = 0 表示近似前沿完全在真实前沿上。

    Args:
        approx_front: 近似帕累托前沿，形状为 (n_approx, n_obj)
        true_front: 真实帕累托前沿，形状为 (n_true, n_obj)
        p: 范数阶数，默认2（欧氏距离）

    Returns:
        GD 值
    """
    approx_front = np.asarray(approx_front, dtype=float)
    true_front = np.asarray(true_front, dtype=float)

    if approx_front.ndim == 1:
        approx_front = approx_front.reshape(1, -1)
    if true_front.ndim == 1:
        true_front = true_front.reshape(1, -1)

    if approx_front.size == 0 or true_front.size == 0:
        return float('inf')

    n_approx = len(approx_front)
    distances = np.zeros(n_approx)

    for i in range(n_approx):
        diff = true_front - approx_front[i]
        dists = np.linalg.norm(diff, ord=p, axis=1)
        distances[i] = np.min(dists)

    gd = (np.sum(distances ** p) / n_approx) ** (1.0 / p)
    return float(gd)


def inverted_generational_distance(approx_front: np.ndarray, true_front: np.ndarray, p: float = 2.0) -> float:
    """计算反转世代距离 (Inverted Generational Distance, IGD)

    IGD 衡量真实前沿到近似前沿的平均距离，同时反映收敛性和多样性。
    值越小越好，IGD = 0 表示近似前沿完全覆盖真实前沿。

    Args:
        approx_front: 近似帕累托前沿，形状为 (n_approx, n_obj)
        true_front: 真实帕累托前沿，形状为 (n_true, n_obj)
        p: 范数阶数，默认2（欧氏距离）

    Returns:
        IGD 值
    """
    return generational_distance(true_front, approx_front, p)


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """判断 a 是否支配 b（最小化问题）"""
    return np.all(a <= b) and np.any(a < b)


def _filter_nondominated(points: np.ndarray) -> np.ndarray:
    """过滤出非支配点集"""
    if len(points) <= 1:
        return points.copy()

    is_nondominated = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not is_nondominated[i]:
            continue
        for j in range(len(points)):
            if i == j or not is_nondominated[j]:
                continue
            if _dominates(points[j], points[i]):
                is_nondominated[i] = False
                break

    return points[is_nondominated]


def hypervolume(approx_front: np.ndarray, reference_point: np.ndarray) -> float:
    """计算超体积指标 (Hypervolume, HV)

    HV 衡量近似前沿和参考点围成的区域体积，同时反映收敛性和多样性。
    值越大越好。

    使用增量式计算方法（WFG 算法的简化版本），适用于低维目标空间。

    Args:
        approx_front: 近似帕累托前沿，形状为 (n_approx, n_obj)
        reference_point: 参考点，形状为 (n_obj,)

    Returns:
        HV 值
    """
    approx_front = np.asarray(approx_front, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if approx_front.ndim == 1:
        approx_front = approx_front.reshape(1, -1)

    if approx_front.size == 0:
        return 0.0

    n_obj = approx_front.shape[1]

    if reference_point.shape != (n_obj,):
        raise ValueError(f"Reference point must have shape ({n_obj},)")

    # 过滤掉被参考点支配的点（最小化问题中，比参考点差的点）
    valid = np.all(approx_front <= reference_point, axis=1)
    front = approx_front[valid]

    if len(front) == 0:
        return 0.0

    # 取非支配子集
    front = _filter_nondominated(front)

    if n_obj == 1:
        return float(reference_point[0] - np.min(front))

    return _hv_2d(front, reference_point) if n_obj == 2 else _hv_incremental(front, reference_point)


def _hv_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """2维目标空间的超体积计算"""
    sorted_front = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev_y = ref_point[1]

    for i in range(len(sorted_front) - 1, -1, -1):
        width = ref_point[0] - sorted_front[i, 0]
        height = prev_y - sorted_front[i, 1]
        hv += width * height
        prev_y = sorted_front[i, 1]

    return float(hv)


def _hv_incremental(front: np.ndarray, ref_point: np.ndarray) -> float:
    """增量式超体积计算（适用于任意维度，但维数越高越慢）"""
    n_obj = front.shape[1]
    hv = 0.0

    # 按第一个目标降序排列
    sorted_front = front[np.argsort(-front[:, 0])]

    for i in range(len(sorted_front)):
        point = sorted_front[i]

        # 计算 exclusive hypervolume
        # 即该点贡献的、不被其他点覆盖的体积
        excl_hv = _exclusive_hv(point, sorted_front[:i], ref_point, n_obj)
        hv += excl_hv

    return float(hv)


def _exclusive_hv(point: np.ndarray, others: np.ndarray, ref_point: np.ndarray, n_obj: int) -> float:
    """计算一个点的排他超体积"""
    # 初始体积是点到参考点的矩形体积
    total_volume = np.prod(ref_point - point)

    if len(others) == 0:
        return total_volume

    # 用 inclusion-exclusion 原理计算被覆盖的部分
    # 简化：使用递归切分方法
    return _sliced_hv(point, others, ref_point, n_obj, 1)


def _sliced_hv(p: np.ndarray, others: np.ndarray, ref: np.ndarray, n_obj: int, dim: int) -> float:
    """递归切分计算超体积"""
    if dim == n_obj - 1:
        # 最后一维，直接计算
        if len(others) == 0:
            return (ref[dim] - p[dim]) * (ref[dim - 1] - p[dim - 1])

        # 过滤出在第 dim-1 维上更优的点
        mask = others[:, dim - 1] <= p[dim - 1]
        relevant = others[mask]

        if len(relevant) == 0:
            return (ref[dim] - p[dim]) * (ref[dim - 1] - p[dim - 1])

        # 按最后一维排序
        sorted_pts = relevant[np.argsort(relevant[:, dim])]
        area = 0.0
        prev_x = ref[dim - 1]
        curr_y = p[dim]

        for pt in sorted_pts:
            if pt[dim] >= ref[dim]:
                break
            if pt[dim] > curr_y:
                area += (prev_x - p[dim - 1]) * (pt[dim] - curr_y)
                curr_y = pt[dim]
            if pt[dim - 1] < prev_x:
                prev_x = pt[dim - 1]

        if curr_y < ref[dim]:
            area += (prev_x - p[dim - 1]) * (ref[dim] - curr_y)

        return area

    # 更高维，递归处理
    if len(others) == 0:
        volume = 1.0
        for d in range(dim - 1, n_obj):
            volume *= (ref[d] - p[d])
        return volume

    # 按第 dim-1 维排序
    mask = others[:, dim - 1] <= p[dim - 1]
    relevant = others[mask]

    if len(relevant) == 0:
        volume = 1.0
        for d in range(dim - 1, n_obj):
            volume *= (ref[d] - p[d])
        return volume

    sorted_pts = relevant[np.argsort(relevant[:, dim - 1])]
    volume = 0.0
    prev_x = p[dim - 1]

    for i in range(len(sorted_pts) - 1, -1, -1):
        pt = sorted_pts[i]
        if pt[dim - 1] < p[dim - 1]:
            sub_hv = _sliced_hv(pt, sorted_pts[i + 1:], ref, n_obj, dim)
            volume += sub_hv * (prev_x - pt[dim - 1])
            prev_x = pt[dim - 1]

    volume += _sliced_hv(p, np.empty((0, n_obj)), ref, n_obj, dim) * (prev_x - p[dim - 1])

    return volume


def spacing(approx_front: np.ndarray) -> float:
    """计算间距指标 (Spacing)

    Spacing 衡量近似前沿上点的分布均匀性。
    值越小越好，Spacing = 0 表示所有相邻点的距离都相等。

    Args:
        approx_front: 近似帕累托前沿，形状为 (n_approx, n_obj)

    Returns:
        Spacing 值
    """
    approx_front = np.asarray(approx_front, dtype=float)

    if approx_front.ndim == 1:
        approx_front = approx_front.reshape(1, -1)

    n_approx = len(approx_front)

    if n_approx <= 1:
        return 0.0

    # 计算每个点到其他点的最小距离
    min_distances = np.zeros(n_approx)

    for i in range(n_approx):
        diff = approx_front - approx_front[i]
        dists = np.linalg.norm(diff, axis=1)
        dists[i] = np.inf
        min_distances[i] = np.min(dists)

    # 计算最小距离的标准差
    mean_dist = np.mean(min_distances)
    if mean_dist == 0:
        return 0.0

    spacing_val = np.sqrt(np.sum((min_distances - mean_dist) ** 2) / (n_approx - 1)) / mean_dist
    return float(spacing_val)


def spread(approx_front: np.ndarray, true_front: Optional[np.ndarray] = None) -> float:
    """计算延展指标 (Spread / Delta)

    Spread 衡量近似前沿的覆盖范围。
    值越小越好，表示分布越均匀且覆盖范围越广。

    对于真实前沿已知的情况，使用真实前沿的极值点来计算；
    否则使用近似前沿自身的极值点。

    Args:
        approx_front: 近似帕累托前沿，形状为 (n_approx, n_obj)
        true_front: 真实帕累托前沿，形状为 (n_true, n_obj)，可选

    Returns:
        Spread 值
    """
    approx_front = np.asarray(approx_front, dtype=float)

    if approx_front.ndim == 1:
        approx_front = approx_front.reshape(1, -1)

    n_approx = len(approx_front)
    n_obj = approx_front.shape[1]

    if n_approx <= 1:
        return 1.0

    if true_front is not None:
        true_front = np.asarray(true_front, dtype=float)
        if true_front.ndim == 1:
            true_front = true_front.reshape(1, -1)
        # 真实前沿的极值点
        true_min = np.min(true_front, axis=0)
        true_max = np.max(true_front, axis=0)
        range_true = true_max - true_min
        range_true[range_true == 0] = 1.0
    else:
        range_true = np.ones(n_obj)

    # 近似前沿的极值点
    approx_min = np.min(approx_front, axis=0)
    approx_max = np.max(approx_front, axis=0)

    # 计算每个点到其他点的最小距离
    min_distances = np.zeros(n_approx)
    for i in range(n_approx):
        diff = approx_front - approx_front[i]
        dists = np.linalg.norm(diff, axis=1)
        dists[i] = np.inf
        min_distances[i] = np.min(dists)

    mean_dist = np.mean(min_distances)

    # 计算两端的距离（边界距离）
    if true_front is not None:
        # 近似前沿极值点到真实前沿极值点的距离
        d_min = np.linalg.norm((approx_min - true_min) / range_true)
        d_max = np.linalg.norm((approx_max - true_max) / range_true)
    else:
        # 使用近似前沿的边界点到最近邻的距离
        d_min = min_distances[np.argmin(approx_front[:, 0])]
        d_max = min_distances[np.argmax(approx_front[:, 0])]

    # Delta 指标
    if (d_min + d_max + (n_approx - 1) * mean_dist) == 0:
        return 0.0

    delta = (d_min + d_max + np.sum(np.abs(min_distances - mean_dist))) / \
            (d_min + d_max + (n_approx - 1) * mean_dist)

    return float(delta)
