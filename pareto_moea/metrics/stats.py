"""统计分析模块"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def wilcoxon_test(x: np.ndarray, y: np.ndarray,
                  alternative: str = 'two-sided') -> Tuple[float, float]:
    """Wilcoxon符号秩检验

    用于比较两组配对样本是否存在显著差异。
    适用于多次独立运行的算法性能对比。

    Args:
        x: 第一组样本，形状为 (n_samples,)
        y: 第二组样本，形状为 (n_samples,)
        alternative: 备择假设，'two-sided' | 'less' | 'greater'

    Returns:
        (statistic, p_value) 检验统计量和p值
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("两组样本长度必须相同")

    if len(x) < 3:
        return np.nan, np.nan

    try:
        result = stats.wilcoxon(x, y, alternative=alternative, zero_method='wilcox')
        return result.statistic, result.pvalue
    except ValueError:
        return np.nan, np.nan


def ranksums_test(x: np.ndarray, y: np.ndarray,
                  alternative: str = 'two-sided') -> Tuple[float, float]:
    """Wilcoxon秩和检验 (Mann-Whitney U检验)

    用于比较两组独立样本是否存在显著差异。
    适用于两组重复运行次数可以不同的算法性能对比。

    Args:
        x: 第一组样本，形状为 (n_samples,)
        y: 第二组样本，形状为 (m_samples,)
        alternative: 备择假设，'two-sided' | 'less' | 'greater'

    Returns:
        (statistic, p_value) 检验统计量和p值
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_valid = x[~np.isnan(x)]
    y_valid = y[~np.isnan(y)]

    if len(x_valid) < 3 or len(y_valid) < 3:
        return np.nan, np.nan

    try:
        result = stats.ranksums(x_valid, y_valid, alternative=alternative)
        return result.statistic, result.pvalue
    except ValueError:
        return np.nan, np.nan


def pairwise_wilcoxon(data: Dict[str, np.ndarray],
                      alternative: str = 'two-sided') -> Tuple[np.ndarray, List[str]]:
    """两两Wilcoxon秩和检验

    对多个算法的结果进行两两比较，生成p值矩阵。

    Args:
        data: 字典，键为算法名，值为该算法多次运行的指标值数组
        alternative: 备择假设

    Returns:
        (p_matrix, labels) p值矩阵和算法名称列表
    """
    labels = list(data.keys())
    n = len(labels)
    p_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            _, p = wilcoxon_test(data[labels[i]], data[labels[j]], alternative)
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    return p_matrix, labels


def significance_level(p_value: float) -> str:
    """根据p值返回显著性水平标记

    Args:
        p_value: p值

    Returns:
        显著性标记：*** (p<0.001), ** (p<0.01), * (p<0.05), ns (不显著)
    """
    if np.isnan(p_value):
        return 'N/A'
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """计算统计量

    Args:
        values: 数值数组

    Returns:
        包含均值、标准差、中位数、最小值、最大值的字典
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'count': 0
        }

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'count': len(values)
    }


def friedman_test(data: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Friedman检验

    非参数检验，用于比较多个相关样本是否存在显著差异。

    Args:
        data: 字典，键为算法名，值为该算法多次运行的指标值数组

    Returns:
        (statistic, p_value)
    """
    labels = list(data.keys())
    if len(labels) < 2:
        return np.nan, np.nan

    arrays = [np.asarray(data[label], dtype=float) for label in labels]
    n_samples = len(arrays[0])

    if any(len(arr) != n_samples for arr in arrays):
        raise ValueError("所有算法的运行次数必须相同")

    if n_samples < 3:
        return np.nan, np.nan

    try:
        result = stats.friedmanchisquare(*arrays)
        return result.statistic, result.pvalue
    except ValueError:
        return np.nan, np.nan


def effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """计算效应量（Cohen's d的非参数版本：r = Z / sqrt(N)）

    Args:
        x: 第一组样本
        y: 第二组样本

    Returns:
        效应量 r
    """
    from scipy.stats import wilcoxon

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y) or len(x) < 3:
        return np.nan

    try:
        diff = x - y
        n = len(diff)

        ranks = stats.rankdata(np.abs(diff))
        signed_ranks = ranks * np.sign(diff)
        w_plus = np.sum(signed_ranks[signed_ranks > 0])
        w_minus = np.sum(np.abs(signed_ranks[signed_ranks < 0]))
        z = (w_plus - w_minus) / np.sqrt(n * (n + 1) * (2 * n + 1) / 6)

        r = np.abs(z) / np.sqrt(n * 2)
        return float(r)
    except Exception:
        return np.nan
