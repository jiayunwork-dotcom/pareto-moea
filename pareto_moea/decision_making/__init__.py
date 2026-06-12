"""决策支持模块

提供帕累托前沿的决策分析工具，包括膝点检测、TOPSIS排序和区域筛选。
"""

from .decision_support import (
    knee_point_detection,
    knee_point_angle_based,
    knee_point_distance_based,
    topsis,
    region_filter,
)

__all__ = [
    "knee_point_detection",
    "knee_point_angle_based",
    "knee_point_distance_based",
    "topsis",
    "region_filter",
]
