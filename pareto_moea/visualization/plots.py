"""多目标优化可视化模块"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Dict, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D


def _check_2d(objectives: np.ndarray) -> None:
    """检查是否为2目标问题"""
    if objectives.shape[1] != 2:
        raise ValueError("2D plot requires exactly 2 objectives")


def _check_3d(objectives: np.ndarray) -> None:
    """检查是否为3目标问题"""
    if objectives.shape[1] != 3:
        raise ValueError("3D plot requires exactly 3 objectives")


def _check_at_least_3d(objectives: np.ndarray) -> None:
    """检查是否至少为3目标问题"""
    if objectives.shape[1] < 3:
        raise ValueError("Parallel coordinates requires at least 3 objectives")


def plot_pareto_front_2d(
    approx_fronts: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    true_front: Optional[np.ndarray] = None,
    objectives: Optional[np.ndarray] = None,
    constraint_violation: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "Pareto Front (2D)",
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    figsize: Tuple[float, float] = (8, 6),
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    show_legend: bool = True,
    true_front_color: str = "red",
    true_front_label: str = "True Pareto Front",
    feasible_color: str = "blue",
    infeasible_color: str = "orange",
    show_grid: bool = True,
) -> Figure:
    """2D帕累托前沿散点图

    支持显示近似前沿、真实前沿、以及完整种群（区分可行/不可行解）。
    支持多算法叠加对比。

    Args:
        approx_fronts: 近似帕累托前沿。可以是：
            - 单个 ndarray，形状 (n_points, 2)
            - ndarray 列表，每个元素是一个算法的前沿
            - 字典，键为算法名称，值为前沿 ndarray
        true_front: 真实帕累托前沿，形状 (n_points, 2)，可选
        objectives: 完整种群目标值，形状 (n_pop, 2)，可选
        constraint_violation: 约束违反量，形状 (n_pop,)，可选。
            与 objectives 一起使用时，用于区分可行/不可行解
        labels: 算法标签列表，当 approx_fronts 是列表时使用
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图表尺寸
        colors: 颜色列表
        markers: 标记样式列表
        show_legend: 是否显示图例
        true_front_color: 真实前沿颜色
        true_front_label: 真实前沿标签
        feasible_color: 可行解颜色（显示完整种群时）
        infeasible_color: 不可行解颜色（显示完整种群时）
        show_grid: 是否显示网格

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    default_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    if colors is None:
        colors = default_colors
    if markers is None:
        markers = default_markers

    fronts_list = []
    names_list = []

    if isinstance(approx_fronts, dict):
        for name, front in approx_fronts.items():
            front = np.asarray(front, dtype=float)
            if front.ndim == 1:
                front = front.reshape(1, -1)
            _check_2d(front)
            fronts_list.append(front)
            names_list.append(name)
    elif isinstance(approx_fronts, list):
        for i, front in enumerate(approx_fronts):
            front = np.asarray(front, dtype=float)
            if front.ndim == 1:
                front = front.reshape(1, -1)
            _check_2d(front)
            fronts_list.append(front)
            if labels is not None and i < len(labels):
                names_list.append(labels[i])
            else:
                names_list.append(f"Algorithm {i + 1}")
    else:
        front = np.asarray(approx_fronts, dtype=float)
        if front.ndim == 1:
            front = front.reshape(1, -1)
        _check_2d(front)
        fronts_list.append(front)
        if labels is not None and len(labels) > 0:
            names_list.append(labels[0])
        else:
            names_list.append("Approximation")

    if objectives is not None:
        objectives = np.asarray(objectives, dtype=float)
        if objectives.ndim == 1:
            objectives = objectives.reshape(1, -1)
        _check_2d(objectives)

        if constraint_violation is not None:
            cv = np.asarray(constraint_violation, dtype=float)
            feasible_mask = cv <= 1e-10
            infeasible_mask = ~feasible_mask

            if np.any(feasible_mask):
                ax.scatter(
                    objectives[feasible_mask, 0],
                    objectives[feasible_mask, 1],
                    c=feasible_color,
                    marker=".",
                    alpha=0.3,
                    label="Feasible Solutions"
                )

            if np.any(infeasible_mask):
                ax.scatter(
                    objectives[infeasible_mask, 0],
                    objectives[infeasible_mask, 1],
                    c=infeasible_color,
                    marker="x",
                    alpha=0.5,
                    label="Infeasible Solutions"
                )
        else:
            ax.scatter(
                objectives[:, 0],
                objectives[:, 1],
                c="gray",
                marker=".",
                alpha=0.3,
                label="All Solutions"
            )

    for i, front in enumerate(fronts_list):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        label = names_list[i] if i < len(names_list) else f"Front {i + 1}"
        ax.scatter(
            front[:, 0],
            front[:, 1],
            c=[color],
            marker=marker,
            s=40,
            label=label,
            zorder=5
        )

    if true_front is not None:
        true_front = np.asarray(true_front, dtype=float)
        if true_front.ndim == 1:
            true_front = true_front.reshape(1, -1)
        _check_2d(true_front)
        sorted_idx = np.argsort(true_front[:, 0])
        sorted_front = true_front[sorted_idx]
        ax.plot(
            sorted_front[:, 0],
            sorted_front[:, 1],
            c=true_front_color,
            linestyle="--",
            linewidth=2,
            label=true_front_label,
            zorder=3
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if show_grid:
        ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    return fig


def plot_pareto_front_3d(
    approx_fronts: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    true_front: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "Pareto Front (3D)",
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    zlabel: str = "Objective 3",
    figsize: Tuple[float, float] = (10, 8),
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    show_legend: bool = True,
    true_front_color: str = "red",
    true_front_label: str = "True Pareto Front",
    show_grid: bool = True,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
) -> Figure:
    """3D帕累托前沿散点图

    使用 mpl_toolkits.mplot3d 绘制，支持鼠标交互旋转。
    支持多算法叠加对比。

    Args:
        approx_fronts: 近似帕累托前沿。可以是：
            - 单个 ndarray，形状 (n_points, 3)
            - ndarray 列表，每个元素是一个算法的前沿
            - 字典，键为算法名称，值为前沿 ndarray
        true_front: 真实帕累托前沿，形状 (n_points, 3)，可选
        labels: 算法标签列表，当 approx_fronts 是列表时使用
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        zlabel: z轴标签
        figsize: 图表尺寸
        colors: 颜色列表
        markers: 标记样式列表
        show_legend: 是否显示图例
        true_front_color: 真实前沿颜色
        true_front_label: 真实前沿标签
        show_grid: 是否显示网格
        elev: 仰角（度），可选
        azim: 方位角（度），可选

    Returns:
        matplotlib Figure 对象
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    default_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    if colors is None:
        colors = default_colors
    if markers is None:
        markers = default_markers

    fronts_list = []
    names_list = []

    if isinstance(approx_fronts, dict):
        for name, front in approx_fronts.items():
            front = np.asarray(front, dtype=float)
            if front.ndim == 1:
                front = front.reshape(1, -1)
            _check_3d(front)
            fronts_list.append(front)
            names_list.append(name)
    elif isinstance(approx_fronts, list):
        for i, front in enumerate(approx_fronts):
            front = np.asarray(front, dtype=float)
            if front.ndim == 1:
                front = front.reshape(1, -1)
            _check_3d(front)
            fronts_list.append(front)
            if labels is not None and i < len(labels):
                names_list.append(labels[i])
            else:
                names_list.append(f"Algorithm {i + 1}")
    else:
        front = np.asarray(approx_fronts, dtype=float)
        if front.ndim == 1:
            front = front.reshape(1, -1)
        _check_3d(front)
        fronts_list.append(front)
        if labels is not None and len(labels) > 0:
            names_list.append(labels[0])
        else:
            names_list.append("Approximation")

    for i, front in enumerate(fronts_list):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        label = names_list[i] if i < len(names_list) else f"Front {i + 1}"
        ax.scatter(
            front[:, 0],
            front[:, 1],
            front[:, 2],
            c=[color],
            marker=marker,
            s=40,
            label=label,
            zorder=5
        )

    if true_front is not None:
        true_front = np.asarray(true_front, dtype=float)
        if true_front.ndim == 1:
            true_front = true_front.reshape(1, -1)
        _check_3d(true_front)
        ax.scatter(
            true_front[:, 0],
            true_front[:, 1],
            true_front[:, 2],
            c=true_front_color,
            marker=".",
            s=20,
            alpha=0.6,
            label=true_front_label,
            zorder=3
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if show_grid:
        ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(loc="best", fontsize=10)

    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    return fig


def plot_parallel_coordinates(
    approx_fronts: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    labels: Optional[List[str]] = None,
    obj_labels: Optional[List[str]] = None,
    title: str = "Parallel Coordinates Plot",
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[List[str]] = None,
    show_legend: bool = True,
    normalize: bool = True,
    alpha: float = 0.5,
    linewidth: float = 1.0,
) -> Figure:
    """平行坐标图

    展示多目标（3目标以上）之间的权衡关系。
    每个目标对应一条纵轴，每条折线代表一个解。

    Args:
        approx_fronts: 近似帕累托前沿。可以是：
            - 单个 ndarray，形状 (n_points, n_obj)
            - ndarray 列表，每个元素是一个算法的前沿
            - 字典，键为算法名称，值为前沿 ndarray
        labels: 算法标签列表，当 approx_fronts 是列表时使用
        obj_labels: 目标标签列表
        title: 图表标题
        figsize: 图表尺寸
        colors: 颜色列表
        show_legend: 是否显示图例
        normalize: 是否归一化目标值到 [0, 1] 区间
        alpha: 线条透明度
        linewidth: 线条宽度

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if colors is None:
        colors = default_colors

    fronts_list = []
    names_list = []

    if isinstance(approx_fronts, dict):
        for name, front in approx_fronts.items():
            front = np.asarray(front, dtype=float)
            if front.ndim == 1:
                front = front.reshape(1, -1)
            _check_at_least_3d(front)
            fronts_list.append(front)
            names_list.append(name)
    elif isinstance(approx_fronts, list):
        for i, front in enumerate(approx_fronts):
            front = np.asarray(front, dtype=float)
            if front.ndim == 1:
                front = front.reshape(1, -1)
            _check_at_least_3d(front)
            fronts_list.append(front)
            if labels is not None and i < len(labels):
                names_list.append(labels[i])
            else:
                names_list.append(f"Algorithm {i + 1}")
    else:
        front = np.asarray(approx_fronts, dtype=float)
        if front.ndim == 1:
            front = front.reshape(1, -1)
        _check_at_least_3d(front)
        fronts_list.append(front)
        if labels is not None and len(labels) > 0:
            names_list.append(labels[0])
        else:
            names_list.append("Approximation")

    n_obj = fronts_list[0].shape[1]

    if obj_labels is None:
        obj_labels = [f"Objective {i + 1}" for i in range(n_obj)]

    all_data = np.vstack(fronts_list)

    if normalize:
        obj_min = np.min(all_data, axis=0)
        obj_max = np.max(all_data, axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0
    else:
        obj_min = np.zeros(n_obj)
        obj_range = np.ones(n_obj)

    x_positions = np.arange(n_obj)

    for i, front in enumerate(fronts_list):
        color = colors[i % len(colors)]
        label = names_list[i] if i < len(names_list) else f"Front {i + 1}"

        normalized_front = (front - obj_min) / obj_range

        for j in range(len(normalized_front)):
            ax.plot(
                x_positions,
                normalized_front[j],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                label=label if j == 0 else ""
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(obj_labels, fontsize=10)
    ax.set_title(title, fontsize=14)

    if normalize:
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylabel("Objective Value", fontsize=12)

    if show_legend:
        ax.legend(loc="best", fontsize=10)

    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_convergence(
    convergence_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    metric_name: str = "Metric Value",
    labels: Optional[List[str]] = None,
    title: str = "Convergence Curve",
    xlabel: str = "Generation",
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    show_legend: bool = True,
    show_grid: bool = True,
    log_scale: bool = False,
    start_gen: int = 0,
) -> Figure:
    """收敛曲线图

    横轴为代数，纵轴为指标值，支持多算法同图对比。

    Args:
        convergence_data: 收敛数据。可以是：
            - 单个 ndarray，形状 (n_gen,)
            - ndarray 列表，每个元素是一个算法的收敛数据
            - 字典，键为算法名称，值为收敛数据 ndarray
        metric_name: 指标名称（用于y轴标签，当ylabel未设置时）
        labels: 算法标签列表，当 convergence_data 是列表时使用
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签，如未设置则使用 metric_name
        figsize: 图表尺寸
        colors: 颜色列表
        linestyles: 线条样式列表
        show_legend: 是否显示图例
        show_grid: 是否显示网格
        log_scale: y轴是否使用对数刻度
        start_gen: 起始代数（从第0代开始计数）

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    default_linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]

    if colors is None:
        colors = default_colors
    if linestyles is None:
        linestyles = default_linestyles

    data_list = []
    names_list = []

    if isinstance(convergence_data, dict):
        for name, data in convergence_data.items():
            data = np.asarray(data, dtype=float)
            data_list.append(data)
            names_list.append(name)
    elif isinstance(convergence_data, list):
        for i, data in enumerate(convergence_data):
            data = np.asarray(data, dtype=float)
            data_list.append(data)
            if labels is not None and i < len(labels):
                names_list.append(labels[i])
            else:
                names_list.append(f"Algorithm {i + 1}")
    else:
        data = np.asarray(convergence_data, dtype=float)
        data_list.append(data)
        if labels is not None and len(labels) > 0:
            names_list.append(labels[0])
        else:
            names_list.append("Algorithm")

    if ylabel is None:
        ylabel = metric_name

    for i, data in enumerate(data_list):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        label = names_list[i] if i < len(names_list) else f"Algorithm {i + 1}"

        n_gen = len(data)
        gens = np.arange(start_gen, start_gen + n_gen)

        ax.plot(
            gens,
            data,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_yscale("log")

    if show_grid:
        ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    return fig


def plot_boxplot(
    data: Union[List[np.ndarray], Dict[str, np.ndarray]],
    labels: Optional[List[str]] = None,
    title: str = "Performance Comparison",
    ylabel: str = "Metric Value",
    figsize: Tuple[float, float] = (8, 6),
    colors: Optional[List[str]] = None,
    show_grid: bool = True,
    show_means: bool = True,
    show_outliers: bool = True,
    patch_artist: bool = True,
    y_log_scale: bool = False,
) -> Figure:
    """箱线图

    展示多次运行的指标统计分布，用于算法性能比较。

    Args:
        data: 指标数据。可以是：
            - ndarray 列表，每个元素是一个算法的多次运行结果
            - 字典，键为算法名称，值为多次运行结果 ndarray
        labels: 算法标签列表，当 data 是列表时使用
        title: 图表标题
        ylabel: y轴标签
        figsize: 图表尺寸
        colors: 颜色列表
        show_grid: 是否显示网格
        show_means: 是否显示均值
        show_outliers: 是否显示离群值
        patch_artist: 是否填充箱体颜色
        y_log_scale: y轴是否使用对数刻度

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if colors is None:
        colors = default_colors

    data_list = []
    names_list = []

    if isinstance(data, dict):
        for name, values in data.items():
            values = np.asarray(values, dtype=float)
            data_list.append(values)
            names_list.append(name)
    elif isinstance(data, list):
        for i, values in enumerate(data):
            values = np.asarray(values, dtype=float)
            data_list.append(values)
            if labels is not None and i < len(labels):
                names_list.append(labels[i])
            else:
                names_list.append(f"Algorithm {i + 1}")
    else:
        values = np.asarray(data, dtype=float)
        data_list.append(values)
        if labels is not None and len(labels) > 0:
            names_list.append(labels[0])
        else:
            names_list.append("Algorithm")

    boxprops = dict(facecolor="white", edgecolor="black", linewidth=1.5)
    medianprops = dict(color="red", linewidth=2)
    meanprops = dict(
        marker="D",
        markerfacecolor="orange",
        markeredgecolor="black",
        markersize=6
    )
    flierprops = dict(
        marker="o",
        markerfacecolor="gray",
        markersize=4,
        alpha=0.6
    )

    bp = ax.boxplot(
        data_list,
        labels=names_list,
        patch_artist=patch_artist,
        showmeans=show_means,
        showfliers=show_outliers,
        boxprops=boxprops,
        medianprops=medianprops,
        meanprops=meanprops,
        flierprops=flierprops
    )

    if patch_artist:
        for i, patch in enumerate(bp["boxes"]):
            color = colors[i % len(colors)]
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if y_log_scale:
        ax.set_yscale("log")

    if show_grid:
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_generation_animation(
    history: Dict[str, List],
    gen_idx: int = 0,
    plot_type: str = "scatter",
    true_front: Optional[np.ndarray] = None,
    title: str = None,
    figsize: Tuple[float, float] = (8, 6),
    feasible_color: str = "blue",
    infeasible_color: str = "orange",
    pareto_color: str = "red",
    show_grid: bool = True,
) -> Figure:
    """返回指定代数的种群图（用于滑块交互）

    可以绘制2D散点图或3D散点图，根据目标数量自动选择。

    Args:
        history: 算法历史记录字典，包含 'objectives' 和 'cv' 键
        gen_idx: 代数索引（从0开始）
        plot_type: 绘图类型，'scatter' 或 'parallel'
        true_front: 真实帕累托前沿，可选
        title: 图表标题，如未设置则自动生成
        figsize: 图表尺寸
        feasible_color: 可行解颜色
        infeasible_color: 不可行解颜色
        pareto_color: 帕累托前沿颜色
        show_grid: 是否显示网格

    Returns:
        matplotlib Figure 对象
    """
    objectives_list = history.get("objectives", [])
    cv_list = history.get("cv", [])

    if len(objectives_list) == 0:
        raise ValueError("No objectives data in history")

    gen_idx = max(0, min(gen_idx, len(objectives_list) - 1))

    objectives = np.asarray(objectives_list[gen_idx], dtype=float)
    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)

    n_obj = objectives.shape[1]

    if len(cv_list) > gen_idx:
        cv = np.asarray(cv_list[gen_idx], dtype=float)
        feasible_mask = cv <= 1e-10
    else:
        feasible_mask = np.ones(len(objectives), dtype=bool)

    from ..utils.pareto_utils import pareto_front

    if np.any(feasible_mask):
        feasible_obj = objectives[feasible_mask]
        pf_indices = pareto_front(feasible_obj, return_indices=True)
        pf_obj = feasible_obj[pf_indices]
    else:
        pf_obj = pareto_front(objectives)

    if title is None:
        title = f"Generation {gen_idx}"

    if plot_type == "parallel" or n_obj >= 4:
        fig, ax = plt.subplots(figsize=figsize)

        x_positions = np.arange(n_obj)
        obj_labels = [f"Obj {i + 1}" for i in range(n_obj)]

        all_min = np.min(objectives, axis=0)
        all_max = np.max(objectives, axis=0)
        all_range = all_max - all_min
        all_range[all_range == 0] = 1.0

        infeasible_mask = ~feasible_mask
        if np.any(infeasible_mask):
            inf_obj = objectives[infeasible_mask]
            inf_norm = (inf_obj - all_min) / all_range
            for j in range(len(inf_norm)):
                ax.plot(
                    x_positions,
                    inf_norm[j],
                    color=infeasible_color,
                    alpha=0.3,
                    linewidth=1.0,
                    label="Infeasible" if j == 0 else ""
                )

        if np.any(feasible_mask):
            feas_obj = objectives[feasible_mask]
            feas_norm = (feas_obj - all_min) / all_range
            for j in range(len(feas_norm)):
                ax.plot(
                    x_positions,
                    feas_norm[j],
                    color=feasible_color,
                    alpha=0.5,
                    linewidth=1.0,
                    label="Feasible" if j == 0 else ""
                )

        pf_norm = (pf_obj - all_min) / all_range
        for j in range(len(pf_norm)):
            ax.plot(
                x_positions,
                pf_norm[j],
                color=pareto_color,
                alpha=1.0,
                linewidth=2.0,
                label="Pareto Front" if j == 0 else ""
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(obj_labels)
        ax.set_ylabel("Normalized Value")
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best", fontsize=10)

        if show_grid:
            ax.grid(True, alpha=0.3, axis="y")

    elif n_obj == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        infeasible_mask = ~feasible_mask
        if np.any(infeasible_mask):
            inf_obj = objectives[infeasible_mask]
            ax.scatter(
                inf_obj[:, 0],
                inf_obj[:, 1],
                inf_obj[:, 2],
                c=infeasible_color,
                marker="x",
                alpha=0.5,
                label="Infeasible"
            )

        if np.any(feasible_mask):
            feas_obj = objectives[feasible_mask]
            ax.scatter(
                feas_obj[:, 0],
                feas_obj[:, 1],
                feas_obj[:, 2],
                c=feasible_color,
                marker="o",
                alpha=0.6,
                label="Feasible"
            )

        ax.scatter(
            pf_obj[:, 0],
            pf_obj[:, 1],
            pf_obj[:, 2],
            c=pareto_color,
            marker="*",
            s=80,
            label="Pareto Front",
            zorder=5
        )

        if true_front is not None:
            true_front = np.asarray(true_front, dtype=float)
            ax.scatter(
                true_front[:, 0],
                true_front[:, 1],
                true_front[:, 2],
                c="gray",
                marker=".",
                s=10,
                alpha=0.3,
                label="True Front"
            )

        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best", fontsize=10)

        if show_grid:
            ax.grid(True, alpha=0.3)

    else:
        fig, ax = plt.subplots(figsize=figsize)

        infeasible_mask = ~feasible_mask
        if np.any(infeasible_mask):
            inf_obj = objectives[infeasible_mask]
            ax.scatter(
                inf_obj[:, 0],
                inf_obj[:, 1],
                c=infeasible_color,
                marker="x",
                alpha=0.5,
                label="Infeasible"
            )

        if np.any(feasible_mask):
            feas_obj = objectives[feasible_mask]
            ax.scatter(
                feas_obj[:, 0],
                feas_obj[:, 1],
                c=feasible_color,
                marker="o",
                alpha=0.6,
                label="Feasible"
            )

        ax.scatter(
            pf_obj[:, 0],
            pf_obj[:, 1],
            c=pareto_color,
            marker="*",
            s=80,
            label="Pareto Front",
            zorder=5
        )

        if true_front is not None:
            true_front = np.asarray(true_front, dtype=float)
            if true_front.ndim == 1:
                true_front = true_front.reshape(1, -1)
            sorted_idx = np.argsort(true_front[:, 0])
            sorted_front = true_front[sorted_idx]
            ax.plot(
                sorted_front[:, 0],
                sorted_front[:, 1],
                c="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="True Front"
            )

        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best", fontsize=10)

        if show_grid:
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_sensitivity_line(
    param_values,
    metric_means,
    metric_stds=None,
    param_name="Parameter",
    title="Parameter Sensitivity Analysis",
    xlabel=None,
    ylabel="Metric Value",
    figsize=(10, 7),
    colors=None,
    markers=None,
    show_grid=True,
    show_legend=True,
    alpha_band=0.2,
):
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    default_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    if colors is None:
        colors = default_colors
    if markers is None:
        markers = default_markers

    if xlabel is None:
        xlabel = param_name

    param_values = np.asarray(param_values)
    metric_names = list(metric_means.keys())

    for i, metric_name in enumerate(metric_names):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        means = np.asarray(metric_means[metric_name], dtype=float)

        ax.plot(
            param_values,
            means,
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2,
            label=metric_name
        )

        if metric_stds is not None and metric_name in metric_stds:
            stds = np.asarray(metric_stds[metric_name], dtype=float)
            ax.fill_between(
                param_values,
                means - stds,
                means + stds,
                color=color,
                alpha=alpha_band,
                label=f"{metric_name} ± Std" if i == 0 else None
            )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if show_grid:
        ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    return fig
