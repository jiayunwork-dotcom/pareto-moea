"""性能指标与统计分析模块"""

from .indicators import (
    generational_distance,
    inverted_generational_distance,
    hypervolume,
    spacing,
    spacing_std,
    spread
)

from .stats import (
    wilcoxon_test,
    ranksums_test,
    pairwise_wilcoxon,
    significance_level,
    compute_statistics
)

gd = generational_distance
igd = inverted_generational_distance
hv = hypervolume
sp = spacing
spacing_sd = spacing_std

__all__ = [
    'generational_distance', 'gd',
    'inverted_generational_distance', 'igd',
    'hypervolume', 'hv',
    'spacing', 'sp',
    'spacing_std', 'spacing_sd',
    'spread',
    'wilcoxon_test',
    'ranksums_test',
    'pairwise_wilcoxon',
    'significance_level',
    'compute_statistics'
]
