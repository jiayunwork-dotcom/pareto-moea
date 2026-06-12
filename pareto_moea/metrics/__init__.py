"""性能指标与统计分析模块"""

from .indicators import (
    generational_distance,
    inverted_generational_distance,
    hypervolume,
    spacing,
    spread
)

from .stats import (
    wilcoxon_test,
    pairwise_wilcoxon,
    significance_level,
    compute_statistics
)

gd = generational_distance
igd = inverted_generational_distance
hv = hypervolume
sp = spacing

__all__ = [
    'generational_distance', 'gd',
    'inverted_generational_distance', 'igd',
    'hypervolume', 'hv',
    'spacing', 'sp',
    'spread',
    'wilcoxon_test',
    'pairwise_wilcoxon',
    'significance_level',
    'compute_statistics'
]
