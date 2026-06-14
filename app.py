"""多目标进化优化实验与帕累托前沿分析平台"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime

from pareto_moea.problems import (
    ZDT1, ZDT2, ZDT3, ZDT4, ZDT6,
    DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7,
    WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9,
    CustomProblem
)

from pareto_moea.algorithms import (
    NSGA2, NSGA3, MOEAD, SPEA2, SMSEMOA
)

from pareto_moea.metrics import (
    gd, igd, hv, spacing, spacing_std, spread,
    pairwise_wilcoxon, significance_level, compute_statistics,
    ranksums_test
)

from pareto_moea.visualization import (
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_parallel_coordinates,
    plot_convergence,
    plot_boxplot,
    plot_generation_animation,
    plot_sensitivity_line,
    plot_convergence_metrics,
    plot_population_scatter,
    plot_convergence_with_warnings,
)

from pareto_moea.decision_making import (
    knee_point_detection,
    topsis,
    region_filter
)

from pareto_moea.experiments import ExperimentHistory, ExperimentRecord

st.set_page_config(
    page_title="帕累托前沿分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROBLEM_MAP = {
    'ZDT系列': {
        'ZDT1': ZDT1, 'ZDT2': ZDT2, 'ZDT3': ZDT3,
        'ZDT4': ZDT4, 'ZDT6': ZDT6
    },
    'DTLZ系列': {
        'DTLZ1': DTLZ1, 'DTLZ2': DTLZ2, 'DTLZ3': DTLZ3,
        'DTLZ4': DTLZ4, 'DTLZ5': DTLZ5, 'DTLZ6': DTLZ6, 'DTLZ7': DTLZ7
    },
    'WFG系列': {
        'WFG1': WFG1, 'WFG2': WFG2, 'WFG3': WFG3,
        'WFG4': WFG4, 'WFG5': WFG5, 'WFG6': WFG6,
        'WFG7': WFG7, 'WFG8': WFG8, 'WFG9': WFG9
    }
}

ALGORITHM_MAP = {
    'NSGA-II': NSGA2,
    'NSGA-III': NSGA3,
    'MOEA/D': MOEAD,
    'SPEA2': SPEA2,
    'SMS-EMOA': SMSEMOA
}

ALGORITHM_PARAMS = {
    '通用参数': [
        {'key': 'pop_size', 'label': '种群大小', 'type': 'int', 'default': 100, 'min': 10, 'max': 1000, 'step': 10},
        {'key': 'n_gen', 'label': '最大代数', 'type': 'int', 'default': 100, 'min': 10, 'max': 5000, 'step': 10},
        {'key': 'crossover_prob', 'label': '交叉概率', 'type': 'float', 'default': 0.9, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'key': 'crossover_eta', 'label': '交叉分布指数', 'type': 'float', 'default': 20.0, 'min': 1.0, 'max': 100.0, 'step': 1.0},
        {'key': 'mutation_prob', 'label': '变异概率', 'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01},
        {'key': 'mutation_eta', 'label': '变异分布指数', 'type': 'float', 'default': 20.0, 'min': 1.0, 'max': 100.0, 'step': 1.0},
    ],
    'NSGA-II': [],
    'NSGA-III': [
        {'key': 'n_divisions', 'label': '参考点划分数', 'type': 'int', 'default': 12, 'min': 2, 'max': 100, 'step': 1},
    ],
    'MOEA/D': [
        {'key': 'n_weights', 'label': '权重向量数', 'type': 'int', 'default': 100, 'min': 10, 'max': 1000, 'step': 10},
        {'key': 'neighbor_size', 'label': '邻域大小', 'type': 'int', 'default': 20, 'min': 2, 'max': 200, 'step': 1},
    ],
    'SPEA2': [
        {'key': 'archive_size', 'label': '归档集大小', 'type': 'int', 'default': 100, 'min': 10, 'max': 1000, 'step': 10},
    ],
    'SMS-EMOA': [
        {'key': 'n_offspring', 'label': '每代子代数', 'type': 'int', 'default': 1, 'min': 1, 'max': 50, 'step': 1},
    ],
}


def init_session_state():
    """初始化session state"""
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = ExperimentHistory()

    if 'current_results' not in st.session_state:
        st.session_state.current_results = {}

    if 'selected_results' not in st.session_state:
        st.session_state.selected_results = []

    if 'running' not in st.session_state:
        st.session_state.running = False

    if 'current_algorithm' not in st.session_state:
        st.session_state.current_algorithm = None

    if 'current_problem' not in st.session_state:
        st.session_state.current_problem = None

    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = {}

    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = None

    if 'sensitivity_running' not in st.session_state:
        st.session_state.sensitivity_running = False

    if 'monitor_running' not in st.session_state:
        st.session_state.monitor_running = False

    if 'monitor_data' not in st.session_state:
        st.session_state.monitor_data = None

    if 'monitor_selected_gen' not in st.session_state:
        st.session_state.monitor_selected_gen = 0

    if 'monitor_template_loaded' not in st.session_state:
        st.session_state.monitor_template_loaded = False

    if 'benchmark_config' not in st.session_state:
        st.session_state.benchmark_config = None

    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = None

    if 'benchmark_running' not in st.session_state:
        st.session_state.benchmark_running = False

    if 'benchmark_progress' not in st.session_state:
        st.session_state.benchmark_progress = 0

    if 'benchmark_pending_start' not in st.session_state:
        st.session_state.benchmark_pending_start = None

    if 'benchmark_history' not in st.session_state:
        st.session_state.benchmark_history = []

    if 'benchmark_selected_history' not in st.session_state:
        st.session_state.benchmark_selected_history = None

    if 'benchmark_metric_weights' not in st.session_state:
        st.session_state.benchmark_metric_weights = {}


def sidebar_problem_config():
    """侧边栏：问题配置"""
    st.sidebar.header("🔧 问题配置")

    problem_type = st.sidebar.selectbox(
        "问题类型",
        ["内置测试函数", "自定义问题"],
        key="problem_type"
    )

    problem = None

    if problem_type == "内置测试函数":
        series = st.sidebar.selectbox("测试函数系列", list(PROBLEM_MAP.keys()))
        prob_name = st.sidebar.selectbox("测试函数", list(PROBLEM_MAP[series].keys()))

        problem_class = PROBLEM_MAP[series][prob_name]

        if series == 'DTLZ系列':
            n_obj = st.sidebar.slider("目标数量", 2, 10, 3, key="dtlz_n_obj")
            k = st.sidebar.slider("距离参数k", 1, 20, 5, key="dtlz_k")
            problem = problem_class(n_obj=n_obj, k=k)
        elif series == 'WFG系列':
            n_obj = st.sidebar.slider("目标数量", 2, 10, 3, key="wfg_n_obj")
            k = st.sidebar.slider("位置参数k", 2, 20, 4, step=2, key="wfg_k")
            problem = problem_class(n_obj=n_obj, k=k)
        else:
            n_var = st.sidebar.slider("变量数量", 2, 100, 30, key="zdt_n_var")
            problem = problem_class(n_var=n_var)

    else:
        st.sidebar.subheader("自定义目标函数")
        n_var = st.sidebar.slider("变量数量", 1, 50, 10, key="custom_n_var")
        n_obj = st.sidebar.slider("目标数量", 2, 10, 2, key="custom_n_obj")

        obj_code = st.sidebar.text_area(
            "目标函数 (Python lambda)",
            value="lambda x: [sum(x**2), sum((x-1)**2)]",
            height=80,
            key="custom_obj_code"
        )

        xl_val = st.sidebar.number_input("变量下界", value=0.0, key="custom_xl")
        xu_val = st.sidebar.number_input("变量上界", value=1.0, key="custom_xu")

        n_constr = st.sidebar.number_input("约束数量", 0, 10, 0, key="custom_n_constr")
        constraints = []
        for i in range(int(n_constr)):
            constr_code = st.sidebar.text_input(
                f"约束 {i+1} (g(x)<=0 形式)",
                value=f"lambda x: x[{i}] - 0.5",
                key=f"custom_constr_{i}"
            )
            try:
                constraints.append(eval(constr_code))
            except:
                pass

        try:
            obj_func = eval(obj_code)
            xl = np.full(n_var, xl_val)
            xu = np.full(n_var, xu_val)
            problem = CustomProblem(
                objective_func=obj_func,
                n_var=n_var,
                n_obj=n_obj,
                xl=xl,
                xu=xu,
                constraints=constraints if constraints else None,
                name="CustomProblem"
            )
        except Exception as e:
            st.sidebar.error(f"目标函数解析错误: {e}")
            problem = None

    if problem is not None:
        st.sidebar.info(
            f"**{problem.name}**\n\n"
            f"- 变量数: {problem.n_var}\n"
            f"- 目标数: {problem.n_obj}\n"
            f"- 约束数: {problem.n_constr}\n"
            f"- 描述: {problem.description}"
        )

    st.session_state.current_problem = problem
    return problem


def sidebar_algorithm_config():
    """侧边栏：算法配置"""
    st.sidebar.header("⚙️ 算法配置")

    algo_name = st.sidebar.selectbox("算法", list(ALGORITHM_MAP.keys()))
    algo_class = ALGORITHM_MAP[algo_name]

    pop_size = st.sidebar.slider("种群大小", 20, 500, 100, key="pop_size")
    n_gen = st.sidebar.slider("最大代数", 50, 2000, 100, key="n_gen")

    crossover_prob = st.sidebar.slider("交叉概率", 0.0, 1.0, 0.9, 0.05, key="cx_prob")
    crossover_eta = st.sidebar.slider("交叉分布指数", 1.0, 50.0, 20.0, 1.0, key="cx_eta")

    mutation_prob = st.sidebar.slider("变异概率", 0.0, 1.0, 0.1, 0.01, key="mut_prob")
    mutation_eta = st.sidebar.slider("变异分布指数", 1.0, 50.0, 20.0, 1.0, key="mut_eta")

    constraint_strategy = st.sidebar.selectbox(
        "约束处理策略",
        ["feasibility_rule", "penalty", "epsilon"],
        format_func=lambda x: {
            "feasibility_rule": "可行性规则",
            "penalty": "罚函数法",
            "epsilon": "ε约束法"
        }[x],
        key="constraint_strategy"
    )

    extra_params = {}

    if algo_name == 'NSGA-III':
        n_divisions = st.sidebar.slider("参考点划分数", 2, 30, 12, key="nsga3_div")
        extra_params['n_divisions'] = n_divisions

    elif algo_name == 'MOEA/D':
        n_weights = st.sidebar.slider("权重向量数", 10, 500, 100, key="moead_weights")
        neighbor_size = st.sidebar.slider("邻域大小", 2, 100, 20, key="moead_neighbor")
        extra_params['n_weights'] = n_weights
        extra_params['neighbor_size'] = neighbor_size

    elif algo_name == 'SPEA2':
        archive_size = st.sidebar.slider("归档集大小", 20, 500, 100, key="spea2_archive")
        extra_params['archive_size'] = archive_size

    elif algo_name == 'SMS-EMOA':
        n_offspring = st.sidebar.slider("每代子代数", 1, 10, 1, key="sms_offspring")
        extra_params['n_offspring'] = n_offspring

    seed = st.sidebar.number_input("随机种子", 0, None, 42, key="algo_seed")

    algorithm = algo_class(
        pop_size=pop_size,
        n_gen=n_gen,
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation_prob=mutation_prob,
        mutation_eta=mutation_eta,
        constraint_strategy=constraint_strategy,
        seed=seed,
        **extra_params
    )

    st.session_state.current_algorithm = algorithm
    return algo_name, algorithm


def problem_info_tab(problem):
    """问题信息标签页"""
    st.header("📚 问题定义")

    if problem is None:
        st.warning("请在侧边栏配置问题")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("基本信息")
        st.write(f"**名称**: {problem.name}")
        st.write(f"**变量数量**: {problem.n_var}")
        st.write(f"**目标数量**: {problem.n_obj}")
        st.write(f"**约束数量**: {problem.n_constr}")
        st.write(f"**描述**: {problem.description}")

        st.subheader("变量范围")
        bounds_df = pd.DataFrame({
            '变量': [f'x{i+1}' for i in range(problem.n_var)],
            '下界': problem.xl,
            '上界': problem.xu
        })
        if problem.n_var > 10:
            st.dataframe(bounds_df.head(10), use_container_width=True, key="df_problem_bounds_head")
            st.caption(f"... 共 {problem.n_var} 个变量")
        else:
            st.dataframe(bounds_df, use_container_width=True, key="df_problem_bounds")

    with col2:
        st.subheader("真实帕累托前沿")
        true_front = problem.pareto_front(n_points=1000)

        if true_front is not None:
            st.success(f"已知真实帕累托前沿 ({len(true_front)} 个点)")

            if problem.n_obj == 2:
                fig = plot_pareto_front_2d(
                    {'真实前沿': true_front},
                    title=f"{problem.name} - 真实帕累托前沿"
                )
                st.pyplot(fig, use_container_width=True)
            elif problem.n_obj == 3:
                fig = plot_pareto_front_3d(
                    {'真实前沿': true_front},
                    title=f"{problem.name} - 真实帕累托前沿"
                )
                st.pyplot(fig, use_container_width=True)
            else:
                fig = plot_parallel_coordinates(
                    {'真实前沿': true_front},
                    title=f"{problem.name} - 真实帕累托前沿"
                )
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("该问题的真实帕累托前沿未知")


def run_optimization_tab():
    """运行优化标签页"""
    st.header("▶️ 运行优化")

    problem = st.session_state.current_problem
    algorithm = st.session_state.current_algorithm

    if problem is None:
        st.warning("请先在侧边栏配置问题")
        return

    if algorithm is None:
        st.warning("请先在侧边栏配置算法")
        return

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        n_runs = st.number_input("独立运行次数", 1, 100, 1, key="n_runs")

    with col2:
        st.markdown("### ")
        start_button = st.button("🚀 开始优化", type="primary", use_container_width=True, disabled=st.session_state.running, key="btn_start_optimize")

    with col3:
        st.markdown("### ")
        stop_button = st.button("⏹️ 终止", use_container_width=True, disabled=not st.session_state.running, key="btn_stop_optimize")

    progress_bar = st.progress(0)
    status_text = st.empty()

    if start_button and not st.session_state.running:
        st.session_state.running = True
        run_optimization(problem, algorithm, n_runs, progress_bar, status_text)

    if stop_button and st.session_state.running:
        if algorithm:
            algorithm.stop()
        st.session_state.running = False
        st.warning("优化已终止")

    if st.session_state.current_results:
        display_results(problem)


def run_optimization(problem, algorithm, n_runs, progress_bar, status_text):
    """运行优化实验"""
    results = []
    algo_name = type(algorithm).__name__

    for run_idx in range(n_runs):
        if not st.session_state.running:
            break

        status_text.info(f"运行 {run_idx + 1}/{n_runs}: 初始化...")

        algo = type(algorithm)(**algorithm.get_params())
        algo.seed = algorithm.seed + run_idx if algorithm.seed is not None else None

        progress_val = run_idx / n_runs
        progress_bar.progress(progress_val)

        def callback(algo, gen, pop, obj, cv):
            if not st.session_state.running:
                algo.stop()
            curr_progress = (run_idx + gen / algo.n_gen) / n_runs
            progress_bar.progress(curr_progress)
            status_text.info(f"运行 {run_idx + 1}/{n_runs} - 第 {gen}/{algo.n_gen} 代")

        algo.set_callback(callback)

        try:
            result = algo.run(problem, verbose=False)
            results.append(result)

            true_front = problem.pareto_front() if problem.pareto_front() is not None else None
            approx_front = result.pareto_front

            metrics = {}
            if true_front is not None and len(true_front) > 0:
                metrics['GD'] = gd(approx_front, true_front)
                metrics['IGD'] = igd(approx_front, true_front)
                metrics['Spacing'] = spacing(approx_front)
                metrics['Spread'] = spread(approx_front, true_front)
            else:
                metrics['Spacing'] = spacing(approx_front)

            if true_front is not None and len(true_front) > 0:
                ref_point = np.maximum(np.max(approx_front, axis=0), np.max(true_front, axis=0)) * 1.1 + 1e-6
            else:
                ref_point = np.max(approx_front, axis=0) * 1.1 + 1e-6
            metrics['HV'] = hv(approx_front, ref_point)

            result.metrics = metrics

            record = ExperimentRecord(
                algorithm_name=algo_name,
                problem_name=problem.name,
                params=algo.get_params(),
                metrics=metrics,
                runtime=result.runtime,
                result=result
            )
            st.session_state.experiment_history.add_record(record)
            st.session_state.experiment_history = st.session_state.experiment_history

        except Exception as e:
            st.error(f"运行错误: {e}")
            continue

    st.session_state.running = False
    new_results = dict(st.session_state.current_results)
    new_results[algo_name] = results
    st.session_state.current_results = new_results

    progress_bar.progress(1.0)
    status_text.success(f"完成！共运行 {len(results)} 次")

    if results:
        st.session_state.batch_results = results
        st.rerun()


def display_results(problem):
    """展示优化结果"""
    st.divider()
    st.subheader("📊 结果展示")

    algo_names = list(st.session_state.current_results.keys())

    if not algo_names:
        return

    selected_algos = st.multiselect(
        "选择要展示的算法",
        algo_names,
        default=algo_names
    )

    data_dict = {}
    for algo in selected_algos:
        results = st.session_state.current_results[algo]
        if results:
            data_dict[algo] = results[0].pareto_front

    true_front = problem.pareto_front() if problem.pareto_front() is not None else None

    if problem.n_obj == 2:
        fig = plot_pareto_front_2d(
            data_dict,
            true_front=true_front,
            title="帕累托前沿对比"
        )
        st.pyplot(fig, use_container_width=True)
    elif problem.n_obj == 3:
        fig = plot_pareto_front_3d(
            data_dict,
            true_front=true_front,
            title="帕累托前沿对比"
        )
        st.pyplot(fig, use_container_width=True)
    else:
        fig = plot_parallel_coordinates(
            data_dict,
            title="平行坐标图"
        )
        st.pyplot(fig, use_container_width=True)

    display_metrics_table(selected_algos)
    display_generation_animation(problem, selected_algos)


def display_metrics_table(selected_algos):
    """展示性能指标表格"""
    st.subheader("📈 性能指标")

    metrics_data = []
    for algo in selected_algos:
        results = st.session_state.current_results.get(algo, [])
        if not results:
            continue

        metric_names = list(results[0].metrics.keys())
        stats = {}
        for metric in metric_names:
            values = [r.metrics.get(metric, np.nan) for r in results]
            stats[metric] = compute_statistics(np.array(values))

        for metric in metric_names:
            s = stats[metric]
            metrics_data.append({
                '算法': algo,
                '指标': metric,
                '均值': f"{s['mean']:.6f}",
                '标准差': f"{s['std']:.6f}",
                '最小值': f"{s['min']:.6f}",
                '最大值': f"{s['max']:.6f}",
                '运行次数': s['count']
            })

    if metrics_data:
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, key="df_optimize_metrics")


def display_generation_animation(problem, selected_algos):
    """展示代际演化动画（滑块控制）"""
    st.subheader("🎬 代际演化")

    if not selected_algos:
        return

    algo = selected_algos[0]
    results = st.session_state.current_results.get(algo, [])
    if not results:
        return

    result = results[0]
    n_gens = len(result.history['objectives']) - 1

    gen = st.slider("选择代数", 0, n_gens, n_gens, key="anim_gen")

    true_front = problem.pareto_front() if problem.pareto_front() is not None else None

    fig = plot_generation_animation(
        result.history,
        gen_idx=gen,
        true_front=true_front,
        title=f"第 {gen} 代 - {algo}"
    )
    st.pyplot(fig, use_container_width=True)


def comparison_tab():
    """算法对比分析标签页"""
    st.header("📊 算法对比分析")

    history = st.session_state.experiment_history
    records = list(history.records)

    if len(records) < 2:
        st.info("请至少运行两次实验以进行对比分析")
        return

    df = history.to_dataframe(flatten=True)

    st.subheader("选择实验")

    algos = df['algorithm_name'].unique().tolist()
    problems = df['problem_name'].unique().tolist()

    col1, col2 = st.columns(2)
    with col1:
        selected_problem = st.selectbox("选择问题", problems)
    with col2:
        selected_algos = st.multiselect("选择算法", algos, default=algos)

    filtered = df[
        (df['problem_name'] == selected_problem) &
        (df['algorithm_name'].isin(selected_algos))
    ]

    if filtered.empty:
        st.warning("没有符合条件的实验记录")
        return

    metric_cols = [c for c in filtered.columns if c.startswith('metric_')]
    metric_names = [c.replace('metric_', '') for c in metric_cols]

    if not metric_names:
        st.warning("没有可用的性能指标")
        return

    selected_metric = st.selectbox("选择指标", metric_names)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 箱线图")
        data = {}
        for algo in selected_algos:
            algo_data = filtered[filtered['algorithm_name'] == algo][f'metric_{selected_metric}'].values
            if len(algo_data) > 0:
                data[algo] = algo_data

        if data:
            fig = plot_boxplot(data, title=f"{selected_metric} 对比")
            st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("🔬 Wilcoxon秩和检验")
        if len(data) >= 2:
            p_matrix, labels = pairwise_wilcoxon(data)

            p_df = pd.DataFrame(p_matrix, index=labels, columns=labels)
            st.dataframe(p_df.style.format('{:.4f}'), use_container_width=True, key="df_wilcoxon_p")

            st.caption("显著性水平: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著")

            sig_matrix = [[significance_level(p) for p in row] for row in p_matrix]
            sig_df = pd.DataFrame(sig_matrix, index=labels, columns=labels)
            st.dataframe(sig_df, use_container_width=True, key="df_wilcoxon_sig")

    st.subheader("📈 收敛曲线")
    if len(selected_algos) >= 1:
        conv_data = {}
        for algo in selected_algos:
            algo_records = filtered[filtered['algorithm_name'] == algo]
            if len(algo_records) > 0:
                record_id = algo_records.iloc[0]['record_id']
                record = history.get_by_id(record_id)
                if record and record.result:
                    true_front = None
                    problem = st.session_state.current_problem
                    if problem and problem.name == selected_problem:
                        true_front = problem.pareto_front()

                    metric_per_gen = []
                    for gen_obj in record.result.history['objectives']:
                        from pareto_moea.utils.pareto_utils import pareto_front
                        approx_pf = pareto_front(gen_obj)
                        if selected_metric == 'HV':
                            if true_front is not None and len(true_front) > 0:
                                ref_point = np.maximum(np.max(approx_pf, axis=0), np.max(true_front, axis=0)) * 1.1 + 1e-6
                            else:
                                ref_point = np.max(approx_pf, axis=0) * 1.1 + 1e-6
                            val = hv(approx_pf, ref_point)
                        elif true_front is not None and selected_metric in ['GD', 'IGD']:
                            if selected_metric == 'GD':
                                val = gd(approx_pf, true_front)
                            else:
                                val = igd(approx_pf, true_front)
                        elif selected_metric == 'Spacing':
                            val = spacing(approx_pf)
                        else:
                            val = np.nan
                        metric_per_gen.append(val)

                    conv_data[algo] = np.array(metric_per_gen)

        if conv_data:
            fig = plot_convergence(conv_data, title=f"{selected_metric} 收敛曲线")
            st.pyplot(fig, use_container_width=True)


def decision_support_tab():
    """决策支持标签页"""
    st.header("🎯 决策支持")

    problem = st.session_state.current_problem
    current_results = st.session_state.current_results

    if not current_results:
        st.info("请先运行优化以获得帕累托前沿")
        return

    algo_names = list(current_results.keys())
    selected_algo = st.selectbox("选择算法", algo_names)

    results = current_results.get(selected_algo, [])
    if not results:
        return

    result = results[0]
    pareto_front = result.pareto_front
    pareto_set = result.pareto_set

    tab1, tab2, tab3 = st.tabs(["膝点检测", "TOPSIS排序", "区域筛选"])

    with tab1:
        st.subheader("🦵 膝点检测")
        st.write("曲率最大的点，代表效益最平衡的折中解")

        method = st.selectbox(
            "检测方法",
            ["angle", "distance"],
            format_func=lambda x: "基于角度" if x == "angle" else "基于距离"
        )

        try:
            knee_idx, knee_obj = knee_point_detection(pareto_front, method=method)

            st.success(f"膝点索引: {knee_idx}")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**目标值**")
                for i in range(len(knee_obj)):
                    st.write(f"f{i+1}: {knee_obj[i]:.6f}")

            with col2:
                st.write("**决策变量**")
                if knee_idx < len(pareto_set):
                    knee_x = pareto_set[knee_idx]
                    for i in range(min(10, len(knee_x))):
                        st.write(f"x{i+1}: {knee_x[i]:.6f}")
                    if len(knee_x) > 10:
                        st.write(f"... 共 {len(knee_x)} 个变量")

            if problem and problem.n_obj == 2:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                          c='blue', alpha=0.6, label='帕累托前沿')
                ax.scatter(knee_obj[0], knee_obj[1],
                          c='red', s=200, marker='*', label='膝点', zorder=5)
                ax.set_xlabel('f1')
                ax.set_ylabel('f2')
                ax.set_title('帕累托前沿 - 膝点检测')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"膝点检测失败: {e}")

    with tab2:
        st.subheader("🏆 TOPSIS排序")

        n_obj = pareto_front.shape[1]

        weights = []
        cols = st.columns(n_obj)
        for i in range(n_obj):
            with cols[i]:
                w = st.slider(f"目标 f{i+1} 权重", 0.0, 1.0, 1.0/n_obj, 0.05, key=f"weight_{i}")
                weights.append(w)

        weights = np.array(weights)
        if np.sum(weights) == 0:
            st.warning("权重之和不能为0")
        else:
            weights = weights / np.sum(weights)

            scores, ranks = topsis(pareto_front, weights=weights, return_ranks=True)

            top_n = st.slider("显示前 N 个解", 1, min(20, len(pareto_front)), 5)

            top_indices = np.argsort(ranks)[:top_n]

            result_df = pd.DataFrame({
                '排名': ranks[top_indices],
                '贴近度': scores[top_indices],
            })
            for i in range(n_obj):
                result_df[f'f{i+1}'] = pareto_front[top_indices, i]

            st.dataframe(result_df, use_container_width=True, key="df_topsis_result")

            if n_obj == 2:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                          c='lightgray', alpha=0.5, label='所有解')
                scatter = ax.scatter(pareto_front[top_indices, 0], pareto_front[top_indices, 1],
                                    c=scores[top_indices], cmap='viridis', s=100, label='Top解', zorder=5)
                ax.set_xlabel('f1')
                ax.set_ylabel('f2')
                ax.set_title('TOPSIS排序结果')
                plt.colorbar(scatter, label='贴近度')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=True)

    with tab3:
        st.subheader("🎯 区域筛选")
        st.write("在目标空间中框选感兴趣的区域")

        n_obj = pareto_front.shape[1]

        col1, col2 = st.columns(2)
        with col1:
            f1_min = st.number_input("f1 最小值", value=float(np.min(pareto_front[:, 0])), key="f1_min")
            f2_min = st.number_input("f2 最小值", value=float(np.min(pareto_front[:, 1])), key="f2_min")
        with col2:
            f1_max = st.number_input("f1 最大值", value=float(np.max(pareto_front[:, 0])), key="f1_max")
            f2_max = st.number_input("f2 最大值", value=float(np.max(pareto_front[:, 1])), key="f2_max")

        lower = np.array([f1_min, f2_min])
        upper = np.array([f1_max, f2_max])

        if n_obj > 2:
            st.info("当前只显示前两个目标的筛选，其他目标不限制")

        try:
            filtered_obj, filtered_x, filtered_idx = region_filter(
                pareto_front,
                lower_bounds=lower,
                upper_bounds=upper,
                decision_variables=pareto_set,
                return_indices=True
            )

            st.success(f"筛选出 {len(filtered_obj)} 个解")

            if len(filtered_obj) > 0:
                if n_obj == 2:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                              c='lightgray', alpha=0.5, label='所有解')
                    ax.scatter(filtered_obj[:, 0], filtered_obj[:, 1],
                              c='green', s=80, label='筛选结果', zorder=5)
                    ax.set_xlabel('f1')
                    ax.set_ylabel('f2')
                    ax.set_title('区域筛选结果')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig, use_container_width=True)

                st.subheader("筛选出的解")
                df = pd.DataFrame({
                    f'f{i+1}': filtered_obj[:, i] for i in range(min(n_obj, 5))
                })
                for i in range(min(5, filtered_x.shape[1])):
                    df[f'x{i+1}'] = filtered_x[:, i]
                st.dataframe(df, use_container_width=True, key="df_region_filter")

        except Exception as e:
            st.error(f"筛选失败: {e}")


def _get_available_params(algo_name):
    """获取指定算法可用的参数列表"""
    params = list(ALGORITHM_PARAMS['通用参数'])
    if algo_name in ALGORITHM_PARAMS:
        params.extend(ALGORITHM_PARAMS[algo_name])
    return params


def _build_param_values(param_meta, start, end, step):
    """根据范围和步长构建参数值列表"""
    if param_meta['type'] == 'int':
        start = int(start)
        end = int(end)
        step = int(step) if step > 0 else 1
        values = list(range(start, end + 1, step))
    else:
        import numpy as np
        values = []
        v = start
        while v <= end + 1e-9:
            values.append(round(v, 6))
            v += step
    return values


def _create_algorithm_with_param(algo_name, algo_class, base_params, param_key, param_value, run_seed=None):
    """创建算法实例，覆盖指定参数"""
    params = dict(base_params)
    params[param_key] = param_value
    if run_seed is not None:
        params['seed'] = run_seed
    extra_keys = set()
    if algo_name == 'NSGA-III' and 'n_divisions' in params:
        extra_keys.add('n_divisions')
    if algo_name == 'MOEA/D':
        extra_keys.update(['n_weights', 'neighbor_size'])
    if algo_name == 'SPEA2' and 'archive_size' in params:
        extra_keys.add('archive_size')
    if algo_name == 'SMS-EMOA' and 'n_offspring' in params:
        extra_keys.add('n_offspring')

    extra = {k: params.pop(k) for k in list(params.keys()) if k in extra_keys}

    try:
        return algo_class(**params, **extra)
    except Exception:
        return algo_class(**params)


def sensitivity_analysis_tab():
    """算法参数灵敏度分析标签页"""
    st.header("🔬 算法参数灵敏度分析")
    st.caption("系统分析算法参数对性能指标的影响，帮助选择最优参数配置")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("⚙️ 参数配置")

        sa_algo_name = st.selectbox(
            "选择算法",
            list(ALGORITHM_MAP.keys()),
            key="sa_algo",
            index=0
        )
        algo_class = ALGORITHM_MAP[sa_algo_name]

        st.divider()
        st.markdown("**问题配置**")

        sa_series = st.selectbox("测试函数系列", list(PROBLEM_MAP.keys()), key="sa_series")
        sa_prob_name = st.selectbox("测试函数", list(PROBLEM_MAP[sa_series].keys()), key="sa_prob")
        problem_class = PROBLEM_MAP[sa_series][sa_prob_name]

        if sa_series == 'DTLZ系列':
            sa_n_obj = st.slider("目标数量", 2, 10, 3, key="sa_dtlz_n_obj")
            sa_k = st.slider("距离参数k", 1, 20, 5, key="sa_dtlz_k")
            sa_problem = problem_class(n_obj=sa_n_obj, k=sa_k)
        elif sa_series == 'WFG系列':
            sa_n_obj = st.slider("目标数量", 2, 10, 3, key="sa_wfg_n_obj")
            sa_k = st.slider("位置参数k", 2, 20, 4, step=2, key="sa_wfg_k")
            sa_problem = problem_class(n_obj=sa_n_obj, k=sa_k)
        else:
            sa_n_var = st.slider("变量数量", 2, 100, 30, key="sa_zdt_n_var")
            sa_problem = problem_class(n_var=sa_n_var)

        st.divider()
        st.markdown("**分析目标参数**")

        available_params = _get_available_params(sa_algo_name)
        param_labels = {p['key']: p['label'] for p in available_params}
        param_meta_map = {p['key']: p for p in available_params}

        sa_param_key = st.selectbox(
            "选择要分析的参数",
            options=list(param_labels.keys()),
            format_func=lambda k: f"{param_labels[k]} ({k})",
            key="sa_param_key"
        )
        selected_meta = param_meta_map[sa_param_key]

        is_float_param = (selected_meta['type'] == 'float')
        _cast = float if is_float_param else int
        _min = _cast(selected_meta['min'])
        _max = _cast(selected_meta['max'])
        _step = _cast(selected_meta['step'])
        _default_end = _cast(min(selected_meta['max'], selected_meta['default'] * 3)) if is_float_param else int(min(selected_meta['max'], int(selected_meta['default'] * 3)))
        _step_min = _cast(selected_meta['step']) if is_float_param else 1

        col_start, col_end, col_step = st.columns(3)
        with col_start:
            sa_start = st.number_input(
                "起始值",
                value=_min,
                min_value=_min,
                max_value=_max,
                step=_step,
                key="sa_start",
                format="%.4f" if is_float_param else "%d"
            )
        with col_end:
            sa_end = st.number_input(
                "终止值",
                value=_default_end,
                min_value=_min,
                max_value=_max,
                step=_step,
                key="sa_end",
                format="%.4f" if is_float_param else "%d"
            )
        with col_step:
            sa_step = st.number_input(
                "步长",
                value=_step,
                min_value=_step_min,
                max_value=_max,
                step=_step,
                key="sa_step",
                format="%.4f" if is_float_param else "%d"
            )

        param_values = _build_param_values(selected_meta, sa_start, sa_end, sa_step)
        st.info(f"参数取值: {param_values} (共 {len(param_values)} 个)")

        sa_n_runs = st.slider(
            "每个参数值重复次数",
            min_value=1, max_value=50, value=5, step=1,
            key="sa_n_runs"
        )

        st.divider()
        st.markdown("**基础算法参数（保持默认）**")

        base_pop_size = st.slider("种群大小", 20, 500, 100, key="sa_pop_size", disabled=(sa_param_key == 'pop_size'))
        base_n_gen = st.slider("最大代数", 20, 2000, 100, key="sa_n_gen", disabled=(sa_param_key == 'n_gen'))
        base_cx_prob = st.slider("交叉概率", 0.0, 1.0, 0.9, 0.05, key="sa_cx_prob", disabled=(sa_param_key == 'crossover_prob'))
        base_cx_eta = st.slider("交叉分布指数", 1.0, 50.0, 20.0, 1.0, key="sa_cx_eta", disabled=(sa_param_key == 'crossover_eta'))
        base_mut_prob = st.slider("变异概率", 0.0, 1.0, 0.1, 0.01, key="sa_mut_prob", disabled=(sa_param_key == 'mutation_prob'))
        base_mut_eta = st.slider("变异分布指数", 1.0, 50.0, 20.0, 1.0, key="sa_mut_eta", disabled=(sa_param_key == 'mutation_eta'))

        base_params = {
            'pop_size': base_pop_size,
            'n_gen': base_n_gen,
            'crossover_prob': base_cx_prob,
            'crossover_eta': base_cx_eta,
            'mutation_prob': base_mut_prob,
            'mutation_eta': base_mut_eta,
            'constraint_strategy': 'feasibility_rule',
            'seed': 42,
        }

        if sa_algo_name == 'NSGA-III':
            base_n_div = st.slider("参考点划分数", 2, 50, 12, key="sa_nsga3_div", disabled=(sa_param_key == 'n_divisions'))
            base_params['n_divisions'] = base_n_div
        elif sa_algo_name == 'MOEA/D':
            base_n_weights = st.slider("权重向量数", 10, 500, 100, key="sa_moead_w", disabled=(sa_param_key == 'n_weights'))
            base_neighbor = st.slider("邻域大小", 2, 100, 20, key="sa_moead_n", disabled=(sa_param_key == 'neighbor_size'))
            base_params['n_weights'] = base_n_weights
            base_params['neighbor_size'] = base_neighbor
        elif sa_algo_name == 'SPEA2':
            base_archive = st.slider("归档集大小", 20, 500, 100, key="sa_spea2_a", disabled=(sa_param_key == 'archive_size'))
            base_params['archive_size'] = base_archive
        elif sa_algo_name == 'SMS-EMOA':
            base_offspring = st.slider("每代子代数", 1, 20, 1, key="sa_sms_o", disabled=(sa_param_key == 'n_offspring'))
            base_params['n_offspring'] = base_offspring

        st.divider()
        col_run, col_stop = st.columns(2)
        with col_run:
            start_analysis = st.button(
                "🚀 开始分析",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.sensitivity_running,
                key="btn_start_sensitivity"
            )
        with col_stop:
            stop_analysis = st.button(
                "⏹️ 终止",
                use_container_width=True,
                disabled=not st.session_state.sensitivity_running,
                key="btn_stop_sensitivity"
            )

    with right_col:
        total_expected = len(param_values) * sa_n_runs
        progress_bar = st.progress(0)
        status_text = st.empty()

        if start_analysis and not st.session_state.sensitivity_running:
            if len(param_values) == 0:
                st.error("参数取值列表为空，请检查范围和步长设置")
            elif sa_start > sa_end:
                st.error("起始值不能大于终止值")
            else:
                st.session_state.sensitivity_running = True
                st.session_state.sensitivity_results = None

                results_by_param = {}
                best_front_by_param = {}

                try:
                    total_completed = 0

                    for p_idx, p_val in enumerate(param_values):
                        if not st.session_state.sensitivity_running:
                            break

                        results_for_p = []
                        best_metric_val = None
                        best_front = None

                        for r_idx in range(sa_n_runs):
                            if not st.session_state.sensitivity_running:
                                break

                            status_text.info(
                                f"📊 参数值 {p_idx + 1}/{len(param_values)} ({param_labels[sa_param_key]}={p_val}), "
                                f"重复 {r_idx + 1}/{sa_n_runs}"
                            )

                            algo = _create_algorithm_with_param(
                                sa_algo_name, algo_class, base_params, sa_param_key, p_val,
                                run_seed=42 + p_idx * 100 + r_idx
                            )

                            curr_progress = total_completed / total_expected
                            progress_bar.progress(curr_progress)

                            def make_callback(pi, ri, total_done):
                                def cb(algo_ref, gen, pop, obj, cv):
                                    if not st.session_state.sensitivity_running:
                                        algo_ref.stop()
                                    gen_progress = (total_done + gen / max(1, algo_ref.n_gen)) / total_expected
                                    progress_bar.progress(gen_progress)
                                return cb

                            algo.set_callback(make_callback(p_idx, r_idx, total_completed))

                            try:
                                result = algo.run(sa_problem, verbose=False)

                                true_front = sa_problem.pareto_front() if sa_problem.pareto_front() is not None else None
                                approx_front = result.pareto_front

                                metrics = {}
                                if true_front is not None and len(true_front) > 0:
                                    metrics['GD'] = gd(approx_front, true_front)
                                    metrics['IGD'] = igd(approx_front, true_front)
                                if true_front is not None and len(true_front) > 0:
                                    ref_point = np.maximum(np.max(approx_front, axis=0), np.max(true_front, axis=0)) * 1.1 + 1e-6
                                else:
                                    ref_point = np.max(approx_front, axis=0) * 1.1 + 1e-6
                                metrics['HV'] = hv(approx_front, ref_point)

                                result.metrics = metrics
                                results_for_p.append(result)

                                if true_front is not None and len(true_front) > 0:
                                    cmp_metric = metrics.get('IGD', metrics.get('GD', metrics['HV']))
                                    is_better = (best_metric_val is None) or (cmp_metric < best_metric_val if 'IGD' in metrics or 'GD' in metrics else cmp_metric > best_metric_val)
                                else:
                                    cmp_metric = metrics['HV']
                                    is_better = (best_metric_val is None) or (cmp_metric > best_metric_val)

                                if is_better:
                                    best_metric_val = cmp_metric
                                    best_front = approx_front

                                record = ExperimentRecord(
                                    algorithm_name=sa_algo_name,
                                    problem_name=sa_problem.name,
                                    params=algo.get_params(),
                                    metrics=metrics,
                                    runtime=result.runtime,
                                    result=result
                                )
                                st.session_state.experiment_history.add_record(record)

                            except Exception as e:
                                st.error(f"运行错误 (参数={p_val}, 重复={r_idx + 1}): {e}")

                            total_completed += 1

                        results_by_param[p_val] = results_for_p
                        best_front_by_param[p_val] = best_front

                    st.session_state.sensitivity_results = {
                        'algorithm': sa_algo_name,
                        'problem': sa_problem.name,
                        'param_key': sa_param_key,
                        'param_label': param_labels[sa_param_key],
                        'param_values': param_values,
                        'results_by_param': results_by_param,
                        'best_front_by_param': best_front_by_param,
                        'n_runs': sa_n_runs,
                        'problem_obj': sa_problem,
                    }
                    progress_bar.progress(1.0)
                    status_text.success("✅ 参数灵敏度分析完成！")
                    st.session_state.sensitivity_running = False
                    st.rerun()

                except Exception as e:
                    st.session_state.sensitivity_running = False
                    st.error(f"分析过程中出错: {e}")

        if stop_analysis and st.session_state.sensitivity_running:
            st.session_state.sensitivity_running = False
            st.warning("⚠️ 分析已终止")

        if not st.session_state.sensitivity_results:
            if not st.session_state.sensitivity_running:
                st.info("👈 请在左侧配置参数后点击\"开始分析\"")
            return

        sa_res = st.session_state.sensitivity_results
        sa_problem_obj = sa_res['problem_obj']

        tab_chart, tab_front, tab_table = st.tabs([
            "📈 灵敏度折线图",
            "🎯 帕累托前沿对比",
            "📋 汇总表格"
        ])

        with tab_chart:
            st.subheader(f"{sa_res['param_label']} 对性能指标的影响")

            param_values = sa_res['param_values']
            metric_list = ['IGD', 'GD', 'HV']
            means_dict = {}
            stds_dict = {}

            for metric_name in metric_list:
                means = []
                stds = []
                has_metric = False
                for pv in param_values:
                    runs = sa_res['results_by_param'].get(pv, [])
                    vals = [r.metrics.get(metric_name, np.nan) for r in runs if metric_name in r.metrics]
                    if vals:
                        stats = compute_statistics(np.array(vals))
                        means.append(stats['mean'])
                        stds.append(stats['std'])
                        has_metric = True
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                if has_metric:
                    means_dict[metric_name] = np.array(means)
                    stds_dict[metric_name] = np.array(stds)

            if means_dict:
                for metric_name in list(means_dict.keys()):
                    st.markdown(f"**{metric_name}**")
                    single_mean = {metric_name: means_dict[metric_name]}
                    single_std = {metric_name: stds_dict[metric_name]}
                    y_label = f"{metric_name} {'(越小越好)' if metric_name in ['IGD', 'GD'] else '(越大越好)'}"
                    fig = plot_sensitivity_line(
                        param_values,
                        single_mean,
                        single_std,
                        param_name=sa_res['param_label'],
                        title=f"{sa_res['algorithm']} on {sa_res['problem']} — {metric_name}",
                        ylabel=y_label,
                        figsize=(10, 5),
                        alpha_band=0.25
                    )
                    st.pyplot(fig, use_container_width=True)
            else:
                st.warning("没有可显示的性能指标数据")

        with tab_front:
            st.subheader(f"各参数值下最佳帕累托前沿叠加对比")

            best_fronts = sa_res['best_front_by_param']
            true_front = sa_problem_obj.pareto_front() if sa_problem_obj.pareto_front() is not None else None

            plot_data = {}
            for pv in param_values:
                front = best_fronts.get(pv)
                if front is not None and len(front) > 0:
                    label = f"{sa_res['param_label']}={pv}"
                    plot_data[label] = front

            if sa_problem_obj.n_obj == 2:
                fig = plot_pareto_front_2d(
                    plot_data,
                    true_front=true_front,
                    title=f"帕累托前沿对比 — {sa_res['algorithm']} on {sa_res['problem']}",
                    show_grid=True
                )
                st.pyplot(fig, use_container_width=True)
            elif sa_problem_obj.n_obj == 3:
                fig = plot_pareto_front_3d(
                    plot_data,
                    true_front=true_front,
                    title=f"帕累托前沿对比 — {sa_res['algorithm']} on {sa_res['problem']}"
                )
                st.pyplot(fig, use_container_width=True)
            else:
                fig = plot_parallel_coordinates(
                    plot_data,
                    title=f"帕累托前沿对比 — {sa_res['algorithm']} on {sa_res['problem']}"
                )
                st.pyplot(fig, use_container_width=True)

        with tab_table:
            st.subheader(f"性能指标汇总 — {sa_res['param_label']}")

            summary_rows = []
            for pv in param_values:
                runs = sa_res['results_by_param'].get(pv, [])
                row = {sa_res['param_label']: pv, '运行次数': len(runs)}
                for metric_name in ['IGD', 'GD', 'HV']:
                    vals = [r.metrics.get(metric_name, np.nan) for r in runs if metric_name in r.metrics]
                    if vals:
                        s = compute_statistics(np.array(vals))
                        row[f'{metric_name}_均值'] = s['mean']
                        row[f'{metric_name}_标准差'] = s['std']
                        row[f'{metric_name}_最小值'] = s['min']
                        row[f'{metric_name}_最大值'] = s['max']
                    else:
                        row[f'{metric_name}_均值'] = np.nan
                        row[f'{metric_name}_标准差'] = np.nan
                        row[f'{metric_name}_最小值'] = np.nan
                        row[f'{metric_name}_最大值'] = np.nan
                summary_rows.append(row)

            summary_df = pd.DataFrame(summary_rows)

            display_cols = [sa_res['param_label'], '运行次数']
            for m in ['IGD', 'GD', 'HV']:
                if f'{m}_均值' in summary_df.columns and not summary_df[f'{m}_均值'].isna().all():
                    display_cols.extend([f'{m}_均值', f'{m}_标准差', f'{m}_最小值', f'{m}_最大值'])

            display_df = summary_df[display_cols].copy()

            format_dict = {}
            for col in display_df.columns:
                if col != sa_res['param_label'] and col != '运行次数':
                    format_dict[col] = '{:.6f}'

            st.dataframe(
                display_df.style.format(format_dict),
                use_container_width=True,
                hide_index=True,
                key="df_sensitivity_summary"
            )

            st.divider()
            col_dl1, col_dl2 = st.columns([1, 3])
            with col_dl1:
                csv_data = display_df.to_csv(index=False, encoding='utf-8-sig')
                fname = f"sensitivity_{sa_res['algorithm']}_{sa_res['problem']}_{sa_res['param_key']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    "📥 导出汇总表格为 CSV",
                    csv_data,
                    fname,
                    "text/csv",
                    use_container_width=True,
                    key="btn_sensitivity_download"
                )
            with col_dl2:
                st.caption("CSV 文件包含各参数值下 IGD/GD/HV 的均值、标准差、最小值和最大值")


def _create_monitor_algorithm(algo_name, algo_class, pop_size, n_gen, seed):
    """创建监控用的算法实例"""
    extra = {}
    if algo_name == 'NSGA-III':
        extra['n_divisions'] = 12
    elif algo_name == 'MOEA/D':
        extra['n_weights'] = pop_size
        extra['neighbor_size'] = 20
    elif algo_name == 'SPEA2':
        extra['archive_size'] = pop_size
    elif algo_name == 'SMS-EMOA':
        extra['n_offspring'] = 1

    return algo_class(
        pop_size=pop_size,
        n_gen=n_gen,
        crossover_prob=0.9,
        crossover_eta=20.0,
        mutation_prob=0.1,
        mutation_eta=20.0,
        constraint_strategy='feasibility_rule',
        seed=seed,
        **extra
    )


def _compute_monitor_metrics(objectives, true_front, ref_point):
    """计算监控用的收敛指标"""
    from pareto_moea.utils.pareto_utils import pareto_front

    feasible_obj = objectives
    approx_pf = pareto_front(feasible_obj)

    metrics = {}

    if true_front is not None and len(true_front) > 0:
        metrics['IGD'] = igd(approx_pf, true_front)
    else:
        metrics['IGD'] = np.nan

    metrics['HV'] = hv(approx_pf, ref_point)
    metrics['Spacing'] = spacing_std(approx_pf)

    return metrics


def _detect_hv_stagnation(generations, hv_values, hv_threshold_pct=0.1, consec_gens=5):
    """检测HV停滞

    当连续consec_gens代HV变化率绝对值小于阈值时，返回停滞起始代数列表和预警信息

    Args:
        generations: 代数列表
        hv_values: HV值列表
        hv_threshold_pct: HV变化率阈值（百分比，如0.1表示0.1%）
        consec_gens: 连续代数要求

    Returns:
        (stagnation_gens, warnings) - 停滞点代数列表，预警信息列表
    """
    stagnation_gens = []
    warnings = []

    if len(hv_values) < consec_gens + 1:
        return stagnation_gens, warnings

    hv_arr = np.asarray(hv_values, dtype=float)
    gen_arr = np.asarray(generations, dtype=int)

    i = 0
    while i < len(hv_arr) - consec_gens:
        all_stable = True
        for j in range(consec_gens):
            if np.isnan(hv_arr[i + j]) or np.isnan(hv_arr[i + j + 1]):
                all_stable = False
                break
            if hv_arr[i + j] == 0:
                change_rate = abs(hv_arr[i + j + 1] - hv_arr[i + j]) / 1e-10 * 100
            else:
                change_rate = abs(hv_arr[i + j + 1] - hv_arr[i + j]) / abs(hv_arr[i + j]) * 100
            if change_rate >= hv_threshold_pct:
                all_stable = False
                break
        if all_stable:
            start_gen = gen_arr[i]
            end_gen = gen_arr[i + consec_gens]
            stagnation_gens.append(int(start_gen))
            warnings.append({
                'type': 'hv_stagnation',
                'start_gen': int(start_gen),
                'end_gen': int(end_gen),
                'message': f"HV在第{start_gen}代~第{end_gen}代期间停滞,建议检查种群多样性"
            })
            i += consec_gens
        else:
            i += 1

    return stagnation_gens, warnings


def _detect_igd_rebound(generations, igd_values, igd_threshold_pct=5.0):
    """检测IGD反弹

    当相邻两个采样点IGD值上升且幅度超过阈值时，返回反弹代数列表和预警信息

    Args:
        generations: 代数列表
        igd_values: IGD值列表
        igd_threshold_pct: IGD反弹阈值（百分比，如5表示5%）

    Returns:
        (rebound_gens, warnings) - 反弹点代数列表，预警信息列表
    """
    rebound_gens = []
    warnings = []

    if len(igd_values) < 2:
        return rebound_gens, warnings

    igd_arr = np.asarray(igd_values, dtype=float)
    gen_arr = np.asarray(generations, dtype=int)

    for i in range(len(igd_arr) - 1):
        prev = igd_arr[i]
        curr = igd_arr[i + 1]
        if np.isnan(prev) or np.isnan(curr):
            continue
        if curr > prev:
            if prev == 0:
                rebound_pct = (curr - prev) / 1e-10 * 100
            else:
                rebound_pct = (curr - prev) / prev * 100
            if rebound_pct >= igd_threshold_pct:
                rebound_gen = int(gen_arr[i + 1])
                rebound_gens.append(rebound_gen)
                warnings.append({
                    'type': 'igd_rebound',
                    'gen': rebound_gen,
                    'rebound_pct': round(rebound_pct, 2),
                    'message': f"IGD在第{rebound_gen}代出现反弹(+{rebound_pct:.2f}%),可能发生早熟收敛"
                })

    return rebound_gens, warnings


def _find_hv_stable_generation(generations, hv_values, hv_threshold_pct=0.1, consec_gens=5):
    """找到HV首次达到稳定的代数（首次连续consec_gens代HV变化率<阈值的起始代数）

    Returns:
        稳定起始代数，若全程未稳定返回None
    """
    if len(hv_values) < consec_gens + 1:
        return None

    hv_arr = np.asarray(hv_values, dtype=float)
    gen_arr = np.asarray(generations, dtype=int)

    for i in range(len(hv_arr) - consec_gens):
        all_stable = True
        for j in range(consec_gens):
            if np.isnan(hv_arr[i + j]) or np.isnan(hv_arr[i + j + 1]):
                all_stable = False
                break
            if hv_arr[i + j] == 0:
                change_rate = abs(hv_arr[i + j + 1] - hv_arr[i + j]) / 1e-10 * 100
            else:
                change_rate = abs(hv_arr[i + j + 1] - hv_arr[i + j]) / abs(hv_arr[i + j]) * 100
            if change_rate >= hv_threshold_pct:
                all_stable = False
                break
        if all_stable:
            return int(gen_arr[i])

    return None


TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


def _ensure_templates_dir():
    """确保templates目录存在"""
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)


def _list_templates():
    """列出所有已保存的模板"""
    _ensure_templates_dir()
    if not os.path.exists(TEMPLATES_DIR):
        return []
    files = [f for f in os.listdir(TEMPLATES_DIR) if f.endswith('.json')]
    return sorted(files)


def _save_template(template_name, config):
    """保存配置模板到JSON文件"""
    _ensure_templates_dir()
    if not template_name.endswith('.json'):
        template_name += '.json'
    filepath = os.path.join(TEMPLATES_DIR, template_name)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return filepath


def _load_template(template_name):
    """从JSON文件加载配置模板"""
    if not template_name.endswith('.json'):
        template_name += '.json'
    filepath = os.path.join(TEMPLATES_DIR, template_name)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _validate_template(config):
    """校验模板配置中的算法名和问题名是否存在

    Returns:
        (validated_config, errors, warnings)
    """
    errors = []
    warnings_list = []

    all_algorithms = list(ALGORITHM_MAP.keys())
    all_problems_flat = []
    for series in PROBLEM_MAP.values():
        all_problems_flat.extend(series.keys())

    validated_algorithms = []
    invalid_algorithms = []
    for algo_name in config.get('algorithms', []):
        if algo_name in all_algorithms:
            validated_algorithms.append(algo_name)
        else:
            invalid_algorithms.append(algo_name)
            errors.append(f"算法 '{algo_name}' 不存在，已跳过")

    if invalid_algorithms:
        for a in invalid_algorithms:
            errors.append(f"❌ 无效算法: {a}")

    problem_name = config.get('problem_name')
    problem_series = config.get('problem_series')
    if problem_name and problem_series:
        if problem_series not in PROBLEM_MAP:
            errors.append(f"❌ 无效问题系列: {problem_series}")
            config.pop('problem_series', None)
            config.pop('problem_name', None)
        elif problem_name not in PROBLEM_MAP[problem_series]:
            errors.append(f"❌ 问题 '{problem_name}' 不存在于系列 '{problem_series}'，已跳过")
            config.pop('problem_name', None)

    config['algorithms'] = validated_algorithms
    return config, errors, warnings_list


def convergence_monitoring_tab():
    """算法收敛性动态监控标签页"""
    st.header("📈 算法收敛性动态监控")
    st.caption("实时监控优化过程中IGD、HV、Spacing等收敛指标的变化趋势 · 支持多算法对比 · 收敛预警 · 参数模板")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("⚙️ 运行配置")

        mon_algo_names = st.multiselect(
            "选择算法（2~5个）",
            list(ALGORITHM_MAP.keys()),
            default=[list(ALGORITHM_MAP.keys())[0], list(ALGORITHM_MAP.keys())[1]] if len(ALGORITHM_MAP) >= 2 else list(ALGORITHM_MAP.keys())[:1],
            key="mon_algos"
        )

        if len(mon_algo_names) < 2:
            st.warning("⚠️ 请至少选择2个算法进行对比")
        elif len(mon_algo_names) > 5:
            st.warning("⚠️ 最多只能选择5个算法")

        st.divider()
        st.markdown("**问题配置**")

        mon_series = st.selectbox("测试函数系列", list(PROBLEM_MAP.keys()), key="mon_series")
        mon_prob_name = st.selectbox("测试函数", list(PROBLEM_MAP[mon_series].keys()), key="mon_prob")
        problem_class = PROBLEM_MAP[mon_series][mon_prob_name]

        if mon_series == 'DTLZ系列':
            mon_n_obj = st.slider("目标数量", 2, 10, 3, key="mon_dtlz_n_obj")
            mon_k = st.slider("距离参数k", 1, 20, 5, key="mon_dtlz_k")
            mon_problem = problem_class(n_obj=mon_n_obj, k=mon_k)
        elif mon_series == 'WFG系列':
            mon_n_obj = st.slider("目标数量", 2, 10, 3, key="mon_wfg_n_obj")
            mon_k = st.slider("位置参数k", 2, 20, 4, step=2, key="mon_wfg_k")
            mon_problem = problem_class(n_obj=mon_n_obj, k=mon_k)
        else:
            mon_n_var = st.slider("变量数量", 2, 100, 30, key="mon_zdt_n_var")
            mon_problem = problem_class(n_var=mon_n_var)

        st.divider()
        st.markdown("**监控参数**")

        mon_n_gen = st.slider("最大代数", 10, 1000, 100, step=10, key="mon_n_gen")
        max_allowed_interval = mon_n_gen // 2

        mon_sample_interval = st.number_input(
            "采样间隔（代）",
            min_value=1,
            max_value=max_allowed_interval,
            value=min(5, max_allowed_interval),
            step=1,
            key="mon_sample_interval"
        )
        mon_pop_size = st.slider("种群大小", 10, 500, 100, step=10, key="mon_pop_size")
        mon_seed = st.number_input("随机种子", 0, None, 42, key="mon_seed")

        is_positive_int = (float(mon_sample_interval) > 0 and
                          float(mon_sample_interval) % 1 == 0)
        interval_valid = is_positive_int and mon_sample_interval <= max_allowed_interval
        algos_valid = 2 <= len(mon_algo_names) <= 5

        if not interval_valid:
            st.warning(
                f"⚠️ 采样间隔必须是正整数且不超过最大代数的一半 ({max_allowed_interval})，"
                f"当前值 {mon_sample_interval} 无效"
            )

        st.divider()
        st.markdown("**收敛预警阈值**")

        mon_hv_threshold = st.number_input(
            "HV停滞阈值（%）",
            min_value=0.001,
            max_value=100.0,
            value=0.1,
            step=0.01,
            format="%.3f",
            key="mon_hv_threshold",
            help="连续5代HV变化率绝对值小于此阈值时触发停滞警告"
        )
        mon_igd_threshold = st.number_input(
            "IGD反弹阈值（%）",
            min_value=0.1,
            max_value=500.0,
            value=5.0,
            step=0.5,
            format="%.1f",
            key="mon_igd_threshold",
            help="相邻采样点IGD上升幅度超过此阈值时触发反弹警告"
        )

        st.divider()

        true_front = mon_problem.pareto_front(n_points=1000)
        has_true_front = true_front is not None and len(true_front) > 0

        if has_true_front:
            st.success(f"✅ 该问题有真实帕累托前沿 ({len(true_front)} 个点)，IGD指标可用")
        else:
            st.info("ℹ️ 该问题真实帕累托前沿未知，IGD指标将不显示")

        st.divider()

        st.markdown("**参数模板**")
        col_tpl1, col_tpl2 = st.columns(2)
        with col_tpl1:
            template_name_input = st.text_input("模板名称", value="", key="mon_tpl_name_input", placeholder="例如: zdt1_benchmark")
            save_tpl_btn = st.button("💾 保存为模板", use_container_width=True, key="btn_save_tpl")
        with col_tpl2:
            available_templates = _list_templates()
            template_options = ["-- 选择模板 --"] + available_templates
            selected_template = st.selectbox("加载模板", template_options, key="mon_tpl_select")
            load_tpl_btn = st.button("📂 加载模板", use_container_width=True, key="btn_load_tpl", disabled=(selected_template == "-- 选择模板 --"))

        if save_tpl_btn:
            if not template_name_input.strip():
                st.error("⚠️ 请输入模板名称")
            else:
                tpl_config = {
                    'algorithms': mon_algo_names,
                    'problem_series': mon_series,
                    'problem_name': mon_prob_name,
                    'n_gen': mon_n_gen,
                    'sample_interval': mon_sample_interval,
                    'pop_size': mon_pop_size,
                    'seed': mon_seed,
                    'hv_threshold_pct': mon_hv_threshold,
                    'igd_threshold_pct': mon_igd_threshold,
                    'problem_params': {}
                }
                if mon_series == 'DTLZ系列':
                    tpl_config['problem_params'] = {'n_obj': mon_n_obj, 'k': mon_k}
                elif mon_series == 'WFG系列':
                    tpl_config['problem_params'] = {'n_obj': mon_n_obj, 'k': mon_k}
                else:
                    tpl_config['problem_params'] = {'n_var': mon_n_var}

                try:
                    _save_template(template_name_input, tpl_config)
                    st.success(f"✅ 模板已保存: {template_name_input}.json")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 保存失败: {e}")

        if load_tpl_btn and selected_template != "-- 选择模板 --":
            try:
                tpl_config = _load_template(selected_template)
                validated_config, errors, warnings_list = _validate_template(tpl_config)

                if errors:
                    for err in errors:
                        st.error(err)

                if 'algorithms' in validated_config and validated_config['algorithms']:
                    st.session_state['mon_algos'] = validated_config['algorithms']
                if 'problem_series' in validated_config:
                    st.session_state['mon_series'] = validated_config['problem_series']
                if 'problem_name' in validated_config:
                    st.session_state['mon_prob'] = validated_config['problem_name']
                if 'n_gen' in validated_config:
                    st.session_state['mon_n_gen'] = validated_config['n_gen']
                if 'sample_interval' in validated_config:
                    st.session_state['mon_sample_interval'] = validated_config['sample_interval']
                if 'pop_size' in validated_config:
                    st.session_state['mon_pop_size'] = validated_config['pop_size']
                if 'seed' in validated_config:
                    st.session_state['mon_seed'] = validated_config['seed']
                if 'hv_threshold_pct' in validated_config:
                    st.session_state['mon_hv_threshold'] = validated_config['hv_threshold_pct']
                if 'igd_threshold_pct' in validated_config:
                    st.session_state['mon_igd_threshold'] = validated_config['igd_threshold_pct']

                problem_params = validated_config.get('problem_params', {})
                if mon_series == 'DTLZ系列':
                    if 'n_obj' in problem_params:
                        st.session_state['mon_dtlz_n_obj'] = problem_params['n_obj']
                    if 'k' in problem_params:
                        st.session_state['mon_dtlz_k'] = problem_params['k']
                elif mon_series == 'WFG系列':
                    if 'n_obj' in problem_params:
                        st.session_state['mon_wfg_n_obj'] = problem_params['n_obj']
                    if 'k' in problem_params:
                        st.session_state['mon_wfg_k'] = problem_params['k']
                else:
                    if 'n_var' in problem_params:
                        st.session_state['mon_zdt_n_var'] = problem_params['n_var']

                st.success(f"✅ 模板已加载: {selected_template}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 加载失败: {e}")

        st.divider()

        col_run, col_stop = st.columns(2)
        with col_run:
            start_monitor = st.button(
                "🚀 启动监控运行",
                type="primary",
                use_container_width=True,
                disabled=(st.session_state.monitor_running or not interval_valid or not algos_valid),
                key="btn_start_monitor"
            )
        with col_stop:
            stop_monitor = st.button(
                "⏹️ 终止",
                use_container_width=True,
                disabled=not st.session_state.monitor_running,
                key="btn_stop_monitor"
            )

    with right_col:
        progress_bar = st.progress(0)
        status_text = st.empty()

        if start_monitor and not st.session_state.monitor_running and interval_valid and algos_valid:
            st.session_state.monitor_running = True
            st.session_state.monitor_data = None

            multi_results = {}
            total_algorithms = len(mon_algo_names)

            for algo_idx, mon_algo_name in enumerate(mon_algo_names):
                if not st.session_state.monitor_running:
                    break

                algo_class = ALGORITHM_MAP[mon_algo_name]
                status_text.info(f"🔄 正在运行算法 {algo_idx + 1}/{total_algorithms}: {mon_algo_name}")

                monitor_generations = []
                monitor_igd = []
                monitor_hv = []
                monitor_spacing = []
                monitor_populations = {}

                algo = _create_monitor_algorithm(
                    mon_algo_name, algo_class, mon_pop_size, mon_n_gen, mon_seed
                )

                true_front_pf = mon_problem.pareto_front() if has_true_front else None

                ref_point = None
                runtime_start = time.time()

                def make_monitor_callback(algo_idx_inner, total_algo_inner, algo_name_inner):
                    def monitor_callback(algo_ref, gen, pop, obj, cv):
                        nonlocal ref_point

                        if not st.session_state.monitor_running:
                            algo_ref.stop()
                            return

                        if gen == 0 or gen % mon_sample_interval == 0 or gen == mon_n_gen:
                            if ref_point is None:
                                if true_front_pf is not None and len(true_front_pf) > 0:
                                    ref_point = np.maximum(
                                        np.max(obj, axis=0),
                                        np.max(true_front_pf, axis=0)
                                    ) * 1.1 + 1e-6
                                else:
                                    ref_point = np.max(obj, axis=0) * 1.1 + 1e-6

                            metrics = _compute_monitor_metrics(obj, true_front_pf, ref_point)

                            monitor_generations.append(gen)
                            monitor_igd.append(metrics['IGD'])
                            monitor_hv.append(metrics['HV'])
                            monitor_spacing.append(metrics['Spacing'])
                            monitor_populations[gen] = obj.copy()

                            algo_progress = algo_idx_inner / total_algo_inner
                            gen_progress = (gen / max(1, mon_n_gen)) / total_algo_inner
                            curr_progress = algo_progress + gen_progress
                            progress_bar.progress(min(curr_progress, 1.0))
                            status_text.info(
                                f"📊 算法 {algo_idx_inner + 1}/{total_algo_inner} [{algo_name_inner}] - "
                                f"第 {gen}/{mon_n_gen} 代 | "
                                f"HV: {metrics['HV']:.4f} | "
                                f"Spacing: {metrics['Spacing']:.6f}"
                                + (f" | IGD: {metrics['IGD']:.6f}" if has_true_front else "")
                            )
                    return monitor_callback

                algo.set_callback(make_monitor_callback(algo_idx, total_algorithms, mon_algo_name))

                try:
                    result = algo.run(mon_problem, verbose=False)
                    runtime_total = time.time() - runtime_start

                    if len(monitor_generations) == 0 or monitor_generations[-1] != mon_n_gen:
                        if ref_point is None:
                            if true_front_pf is not None and len(true_front_pf) > 0:
                                ref_point = np.maximum(
                                    np.max(result.final_objectives, axis=0),
                                    np.max(true_front_pf, axis=0)
                                ) * 1.1 + 1e-6
                            else:
                                ref_point = np.max(result.final_objectives, axis=0) * 1.1 + 1e-6

                        final_metrics = _compute_monitor_metrics(
                            result.final_objectives, true_front_pf, ref_point
                        )
                        monitor_generations.append(mon_n_gen)
                        monitor_igd.append(final_metrics['IGD'])
                        monitor_hv.append(final_metrics['HV'])
                        monitor_spacing.append(final_metrics['Spacing'])
                        monitor_populations[mon_n_gen] = result.final_objectives.copy()

                    hv_stag_gens, hv_warnings = _detect_hv_stagnation(
                        monitor_generations, monitor_hv,
                        hv_threshold_pct=mon_hv_threshold, consec_gens=5
                    )
                    igd_reb_gens, igd_warnings = _detect_igd_rebound(
                        monitor_generations, monitor_igd,
                        igd_threshold_pct=mon_igd_threshold
                    )

                    stable_gen = _find_hv_stable_generation(
                        monitor_generations, monitor_hv,
                        hv_threshold_pct=mon_hv_threshold, consec_gens=5
                    )

                    multi_results[mon_algo_name] = {
                        'algorithm': mon_algo_name,
                        'problem': mon_problem.name,
                        'n_gen': mon_n_gen,
                        'pop_size': mon_pop_size,
                        'sample_interval': mon_sample_interval,
                        'seed': mon_seed,
                        'generations': monitor_generations,
                        'igd': monitor_igd,
                        'hv': monitor_hv,
                        'spacing': monitor_spacing,
                        'populations': monitor_populations,
                        'true_front': true_front_pf,
                        'has_true_front': has_true_front,
                        'problem_obj': mon_problem,
                        'n_obj': mon_problem.n_obj,
                        'runtime': runtime_total,
                        'hv_stagnation_gens': hv_stag_gens,
                        'hv_warnings': hv_warnings,
                        'igd_rebound_gens': igd_reb_gens,
                        'igd_warnings': igd_warnings,
                        'hv_stable_generation': stable_gen,
                        'final_hv': monitor_hv[-1] if len(monitor_hv) > 0 else np.nan,
                        'final_igd': monitor_igd[-1] if len(monitor_igd) > 0 and has_true_front else np.nan,
                        'final_spacing': monitor_spacing[-1] if len(monitor_spacing) > 0 else np.nan,
                    }

                    record = ExperimentRecord(
                        algorithm_name=mon_algo_name,
                        problem_name=mon_problem.name,
                        params=algo.get_params(),
                        metrics={
                            'IGD': monitor_igd[-1] if has_true_front and len(monitor_igd) > 0 else np.nan,
                            'HV': monitor_hv[-1] if len(monitor_hv) > 0 else np.nan,
                            'Spacing': monitor_spacing[-1] if len(monitor_spacing) > 0 else np.nan
                        },
                        runtime=runtime_total,
                        result=result
                    )
                    st.session_state.experiment_history.add_record(record)

                except Exception as e:
                    runtime_total = time.time() - runtime_start
                    import traceback
                    error_detail = traceback.format_exc()

                    multi_results[mon_algo_name] = {
                        'algorithm': mon_algo_name,
                        'problem': mon_problem.name,
                        'n_gen': mon_n_gen,
                        'pop_size': mon_pop_size,
                        'sample_interval': mon_sample_interval,
                        'seed': mon_seed,
                        'generations': list(monitor_generations),
                        'igd': list(monitor_igd),
                        'hv': list(monitor_hv),
                        'spacing': list(monitor_spacing),
                        'populations': dict(monitor_populations),
                        'true_front': true_front_pf,
                        'has_true_front': has_true_front,
                        'problem_obj': mon_problem,
                        'n_obj': mon_problem.n_obj,
                        'runtime': runtime_total,
                        'hv_stagnation_gens': [],
                        'hv_warnings': [],
                        'igd_rebound_gens': [],
                        'igd_warnings': [],
                        'hv_stable_generation': None,
                        'final_hv': monitor_hv[-1] if len(monitor_hv) > 0 else np.nan,
                        'final_igd': monitor_igd[-1] if len(monitor_igd) > 0 and has_true_front else np.nan,
                        'final_spacing': monitor_spacing[-1] if len(monitor_spacing) > 0 else np.nan,
                        'failed': True,
                        'error_message': str(e),
                        'error_detail': error_detail,
                    }

                    status_text.error(f"❌ 算法 {mon_algo_name} 运行出错: {e}")
                    st.error(f"**{mon_algo_name}** 运行出错: `{e}`")

            completed_algo_names = list(multi_results.keys())
            st.session_state.monitor_data = {
                'algorithms': completed_algo_names,
                'multi_results': multi_results,
                'problem': mon_problem.name,
                'problem_obj': mon_problem,
                'n_obj': mon_problem.n_obj,
                'has_true_front': has_true_front,
                'hv_threshold_pct': mon_hv_threshold,
                'igd_threshold_pct': mon_igd_threshold,
            }

            failed_algos = [an for an, res in multi_results.items() if res.get('failed')]
            if failed_algos:
                progress_bar.progress(1.0)
                status_text.warning(
                    f"⚠️ 运行完成，但 {len(failed_algos)} 个算法失败: {', '.join(failed_algos)}。"
                    f"已完成 {len(completed_algo_names) - len(failed_algos)} 个算法的结果已展示。"
                )
            else:
                progress_bar.progress(1.0)
                status_text.success(f"✅ 所有 {len(completed_algo_names)} 个算法监控运行完成！")
            st.session_state.monitor_running = False
            st.rerun()

        if stop_monitor and st.session_state.monitor_running:
            st.session_state.monitor_running = False
            status_text.warning("⚠️ 监控已终止")

        if not st.session_state.monitor_data:
            if not st.session_state.monitor_running:
                st.info("👈 请在左侧配置参数后点击\"启动监控运行\"")
            return

        mon_data = st.session_state.monitor_data
        multi_results = mon_data['multi_results']
        algo_names = mon_data['algorithms']

        failed_algos = [an for an in algo_names if multi_results[an].get('failed')]
        success_algos = [an for an in algo_names if not multi_results[an].get('failed')]

        if failed_algos:
            st.subheader("❌ 运行失败算法")
            for an in failed_algos:
                res = multi_results[an]
                with st.expander(f"❌ {an} - 运行失败", expanded=True):
                    st.error(f"**错误信息**: `{res.get('error_message', '未知错误')}`")
                    partial_gen_count = len(res['generations'])
                    if partial_gen_count > 0:
                        st.info(f"该算法在失败前已采集 {partial_gen_count} 个采样点（第 {res['generations'][0]}~{res['generations'][-1]} 代）")
                    with st.expander("查看详细错误堆栈"):
                        st.code(res.get('error_detail', ''), language='traceback')

        all_warnings = []
        for an in success_algos:
            res = multi_results[an]
            for w in res.get('hv_warnings', []):
                all_warnings.append((an, w))
            for w in res.get('igd_warnings', []):
                all_warnings.append((an, w))

        if all_warnings:
            st.subheader("⚠️ 收敛预警")
            for algo_n, w in all_warnings:
                if w['type'] == 'hv_stagnation':
                    st.warning(f"**{algo_n}**: {w['message']}")
                elif w['type'] == 'igd_rebound':
                    st.error(f"**{algo_n}**: {w['message']}")

        if not success_algos:
            st.error("❌ 所有算法均运行失败，无可用数据展示")
            return

        display_algo_names = success_algos

        st.subheader("📊 收敛指标对比曲线")

        hv_data = {}
        igd_data = {}
        spacing_data = {}
        hv_stag_points = {}
        igd_reb_points = {}

        for an in display_algo_names:
            res = multi_results[an]
            hv_data[an] = res['hv']
            spacing_data[an] = res['spacing']
            if res['has_true_front']:
                igd_data[an] = res['igd']
            hv_stag_points[an] = res['hv_stagnation_gens']
            igd_reb_points[an] = res['igd_rebound_gens']

        gens = multi_results[display_algo_names[0]]['generations']

        fig_hv = plot_convergence_with_warnings(
            hv_data,
            metric_name="HV (Hypervolume)",
            title="HV 收敛曲线对比 (越大越好)",
            xlabel="代数 (Generation)",
            ylabel="HV 值",
            figsize=(10, 5),
            hv_stagnation_points=hv_stag_points,
            start_gen=0
        )
        st.pyplot(fig_hv, use_container_width=True)

        fig_spacing = plot_convergence_with_warnings(
            spacing_data,
            metric_name="Spacing",
            title="Spacing 收敛曲线对比 (越小越好)",
            xlabel="代数 (Generation)",
            ylabel="Spacing 值",
            figsize=(10, 5),
            start_gen=0
        )
        st.pyplot(fig_spacing, use_container_width=True)

        if mon_data['has_true_front'] and igd_data:
            fig_igd = plot_convergence_with_warnings(
                igd_data,
                metric_name="IGD (Inverted Generational Distance)",
                title="IGD 收敛曲线对比 (越小越好)",
                xlabel="代数 (Generation)",
                ylabel="IGD 值",
                figsize=(10, 5),
                igd_rebound_points=igd_reb_points,
                start_gen=0
            )
            st.pyplot(fig_igd, use_container_width=True)

        st.divider()

        st.subheader("🎯 目标空间种群分布对比（多列布局）")

        if len(gens) > 0:
            gen_labels = [f"第 {g} 代" for g in gens]

            col_slider1, col_slider2 = st.columns([3, 1])
            with col_slider1:
                selected_idx = st.select_slider(
                    "选择代数",
                    options=list(range(len(gens))),
                    value=len(gens) - 1,
                    format_func=lambda x: gen_labels[x],
                    key="mon_gen_slider"
                )
            with col_slider2:
                st.info(f"当前: {gen_labels[selected_idx]}")

            selected_gen = gens[selected_idx]

            n_algos = len(display_algo_names)
            scatter_cols = st.columns(n_algos)

            for i, an in enumerate(display_algo_names):
                res = multi_results[an]
                with scatter_cols[i]:
                    st.markdown(f"**{an}**")
                    current_pop = res['populations'].get(selected_gen)
                    true_front_pf = res.get('true_front')

                    if current_pop is not None:
                        fig_scatter = plot_population_scatter(
                            current_pop,
                            true_front=true_front_pf,
                            title=f"第 {selected_gen} 代",
                            figsize=(6, 5),
                            point_color='#1f77b4',
                            true_front_color='#d62728'
                        )
                        st.pyplot(fig_scatter, use_container_width=True)

                        pop_metrics = {}
                        if res['has_true_front'] and len(res['igd']) > selected_idx:
                            pop_metrics['IGD'] = f"{res['igd'][selected_idx]:.6f}"
                        if len(res['hv']) > selected_idx:
                            pop_metrics['HV'] = f"{res['hv'][selected_idx]:.6f}"
                        if len(res['spacing']) > selected_idx:
                            pop_metrics['Spacing'] = f"{res['spacing'][selected_idx]:.6f}"

                        st.caption(" | ".join([f"{k}: {v}" for k, v in pop_metrics.items()]))
                    else:
                        st.info("无数据")

        st.divider()

        st.subheader("📈 收敛速度对比表")

        speed_rows = []
        for an in algo_names:
            res = multi_results[an]
            if res.get('failed'):
                speed_rows.append({
                    '算法名': an,
                    '达到HV稳定的代数': '运行失败',
                    '最终HV值': np.nan,
                    '最终IGD值': np.nan,
                    '最终Spacing值': np.nan,
                    '总耗时(秒)': round(res.get('runtime', 0.0), 3),
                })
            else:
                stable_gen = res.get('hv_stable_generation')
                speed_rows.append({
                    '算法名': an,
                    '达到HV稳定的代数': stable_gen if stable_gen is not None else '未收敛',
                    '最终HV值': res.get('final_hv', np.nan),
                    '最终IGD值': res.get('final_igd', np.nan),
                    '最终Spacing值': res.get('final_spacing', np.nan),
                    '总耗时(秒)': round(res.get('runtime', 0.0), 3),
                })

        df_speed = pd.DataFrame(speed_rows)

        speed_format_cols = ['最终HV值', '最终IGD值', '最终Spacing值', '总耗时(秒)']
        speed_format_dict = {}
        for col in speed_format_cols:
            speed_format_dict[col] = '{:.6f}'

        st.dataframe(
            df_speed.style.format(speed_format_dict),
            use_container_width=True,
            hide_index=True,
            key="df_speed_comparison"
        )

        st.divider()

        st.subheader("📋 收敛数据表格")

        for an in algo_names:
            res = multi_results[an]
            if res.get('failed'):
                tag = "❌"
            else:
                tag = "📊"
            with st.expander(f"{tag} {an} - 详细数据"):
                if res.get('failed'):
                    st.error(f"该算法运行失败: `{res.get('error_message', '未知错误')}`")
                    if len(res['generations']) > 0:
                        st.info(f"失败前已采集 {len(res['generations'])} 个采样点，以下为部分数据")
                    else:
                        continue

                df_data = {
                    '代数': res['generations'],
                    'HV': res['hv'],
                    'Spacing': res['spacing']
                }
                if res['has_true_front']:
                    df_data['IGD'] = res['igd']

                df_monitor = pd.DataFrame(df_data)

                if res['has_true_front']:
                    display_cols = ['代数', 'IGD', 'HV', 'Spacing']
                else:
                    display_cols = ['代数', 'HV', 'Spacing']

                format_dict = {}
                for col in display_cols:
                    if col != '代数':
                        format_dict[col] = '{:.6f}'

                st.dataframe(
                    df_monitor[display_cols].style.format(format_dict),
                    use_container_width=True,
                    hide_index=True,
                    key=f"df_monitor_data_{an}"
                )

                csv_data = df_monitor[display_cols].to_csv(index=False, encoding='utf-8-sig')
                fname = f"convergence_{an}_{res['problem']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    "📥 导出为 CSV",
                    csv_data,
                    fname,
                    "text/csv",
                    use_container_width=True,
                    key=f"btn_monitor_download_{an}"
                )


def history_tab():
    """实验记录标签页"""
    st.header("📋 实验记录")

    history = st.session_state.experiment_history
    records = list(history.records)

    if not records:
        st.info("暂无实验记录")
        return

    st.subheader("实验历史")

    df = history.to_dataframe(flatten=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_algo = st.multiselect("按算法筛选", df['algorithm_name'].unique().tolist(),
                                     default=df['algorithm_name'].unique().tolist())
    with col2:
        filter_problem = st.multiselect("按问题筛选", df['problem_name'].unique().tolist(),
                                        default=df['problem_name'].unique().tolist())
    with col3:
        filter_metric = st.selectbox("排序指标",
                                     ['timestamp'] + [c for c in df.columns if c.startswith('metric_')],
                                     index=0)

    filtered = df[
        df['algorithm_name'].isin(filter_algo) &
        df['problem_name'].isin(filter_problem)
    ]

    if filter_metric != 'timestamp':
        filtered = filtered.sort_values(filter_metric)

    st.dataframe(filtered, use_container_width=True, hide_index=True, key="df_history_records")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 导出为 CSV", use_container_width=True, key="btn_history_export"):
            csv = filtered.to_csv(index=False)
            st.download_button(
                "下载 CSV 文件",
                csv,
                f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True,
                key="btn_history_download"
            )

    with col2:
        if st.button("🗑️ 清空所有记录", use_container_width=True, type="secondary", key="btn_history_clear"):
            history.clear()
            st.rerun()

    st.subheader("统计汇总")
    metric_cols = [c for c in df.columns if c.startswith('metric_')]
    if metric_cols:
        summary = history.summary()
        st.dataframe(summary, use_container_width=True, key="df_history_summary")


def _create_problem_by_name(problem_name):
    """根据问题名称创建问题实例"""
    for series, problems in PROBLEM_MAP.items():
        if problem_name in problems:
            problem_class = problems[problem_name]
            if series == 'DTLZ系列':
                return problem_class(n_obj=3, k=5)
            elif series == 'WFG系列':
                return problem_class(n_obj=3, k=4)
            else:
                return problem_class(n_var=30)
    return None


def _create_algorithm_by_name(algo_name, pop_size, n_gen, seed):
    """根据算法名称创建算法实例"""
    algo_class = ALGORITHM_MAP[algo_name]

    base_params = {
        'pop_size': pop_size,
        'n_gen': n_gen,
        'crossover_prob': 0.9,
        'crossover_eta': 20.0,
        'mutation_prob': 0.1,
        'mutation_eta': 20.0,
        'constraint_strategy': 'feasibility_rule',
        'seed': seed,
    }

    extra = {}
    if algo_name == 'NSGA-III':
        extra['n_divisions'] = 12
    elif algo_name == 'MOEA/D':
        extra['n_weights'] = pop_size
        extra['neighbor_size'] = 20
    elif algo_name == 'SPEA2':
        extra['archive_size'] = pop_size
    elif algo_name == 'SMS-EMOA':
        extra['n_offspring'] = 1

    return algo_class(**base_params, **extra)


def _compute_benchmark_metrics(approx_front, problem, selected_metrics):
    """计算指定的评测指标"""
    metrics = {}
    true_front = problem.pareto_front() if problem.pareto_front() is not None else None

    if 'IGD' in selected_metrics and true_front is not None and len(true_front) > 0:
        metrics['IGD'] = igd(approx_front, true_front)

    if 'Spacing' in selected_metrics:
        metrics['Spacing'] = spacing(approx_front)

    if 'HV' in selected_metrics:
        if true_front is not None and len(true_front) > 0:
            ref_point = np.maximum(np.max(approx_front, axis=0), np.max(true_front, axis=0)) * 1.1 + 1e-6
        else:
            ref_point = np.max(approx_front, axis=0) * 1.1 + 1e-6
        metrics['HV'] = hv(approx_front, ref_point)

    return metrics


def _compute_benchmark_stats(results_by_repeat, selected_metrics):
    """计算统计值（均值和标准差）"""
    stats = {}
    for metric in selected_metrics:
        values = [r['metrics'].get(metric, np.nan) for r in results_by_repeat if r['error'] is None]
        if values:
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }
        else:
            stats[metric] = {
                'mean': np.nan,
                'std': np.nan,
                'count': 0
            }
    return stats


def _compute_benchmark_rankings(benchmark_results, selected_metrics, metric_weights=None):
    """计算综合排名（支持加权）

    Args:
        benchmark_results: 评测结果字典
        selected_metrics: 选中的指标列表
        metric_weights: 指标权重字典，None表示全部权重为1.0
    """
    algorithms = list(benchmark_results['results'].keys())
    problems = list(benchmark_results['results'][algorithms[0]].keys()) if algorithms else []

    if metric_weights is None:
        metric_weights = {m: 1.0 for m in selected_metrics}

    rankings = {metric: {} for metric in selected_metrics}
    overall_scores = {algo: 0 for algo in algorithms}

    for metric in selected_metrics:
        weight = metric_weights.get(metric, 1.0)
        for problem in problems:
            algo_values = []
            for algo in algorithms:
                stats = benchmark_results['results'][algo][problem]['stats']
                if metric in stats and stats[metric]['count'] > 0:
                    algo_values.append((algo, stats[metric]['mean']))

            if len(algo_values) < 2:
                continue

            if metric in ['IGD', 'Spacing']:
                algo_values.sort(key=lambda x: x[1])
            else:
                algo_values.sort(key=lambda x: -x[1])

            for rank, (algo, val) in enumerate(algo_values):
                if metric not in rankings:
                    rankings[metric] = {}
                if problem not in rankings[metric]:
                    rankings[metric][problem] = {}
                rankings[metric][problem][algo] = rank + 1
                overall_scores[algo] += (rank + 1) * weight

    overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1])

    return rankings, overall_ranking, overall_scores


def _format_mean_std(mean_val, std_val):
    """格式化均值±标准差显示"""
    if np.isnan(mean_val) or np.isnan(std_val):
        return "N/A"
    return f"{mean_val:.4f}(±{std_val:.4f})"


def _compute_benchmark_significance(benchmark_results, selected_metrics):
    """计算显著性检验标记和胜出次数

    使用 Wilcoxon 秩和检验 (ranksums) 两两比较同一问题同一指标下各算法的重复运行值。
    p<0.05 标注 †，p<0.01 标注 ‡。

    Returns:
        significance_markers: dict[metric][problem][algo] -> str ('' or '†' or '‡')
        win_counts: dict[algo] -> int (每个算法显著胜出的总次数)
    """
    algorithms = list(benchmark_results['results'].keys())
    problems = (list(benchmark_results['results'][algorithms[0]].keys())
                if algorithms else [])

    significance_markers = {m: {p: {a: '' for a in algorithms} for p in problems}
                            for m in selected_metrics}
    win_counts = {a: 0 for a in algorithms}

    for metric in selected_metrics:
        lower_is_better = metric in ['IGD', 'Spacing']

        for problem in problems:
            algo_values = {}
            for algo in algorithms:
                repeats = benchmark_results['results'][algo][problem]['repeats']
                vals = [r['metrics'].get(metric, np.nan) for r in repeats
                        if r['error'] is None and metric in r['metrics']]
                if vals:
                    algo_values[algo] = np.array(vals, dtype=float)

            algo_list = list(algo_values.keys())
            for i in range(len(algo_list)):
                for j in range(i + 1, len(algo_list)):
                    a1, a2 = algo_list[i], algo_list[j]
                    v1, v2 = algo_values[a1], algo_values[a2]

                    if len(v1) < 3 or len(v2) < 3:
                        continue

                    try:
                        _, p_two = ranksums_test(v1, v2, alternative='two-sided')
                    except Exception:
                        continue

                    if np.isnan(p_two):
                        continue

                    m1, m2 = np.nanmean(v1), np.nanmean(v2)
                    if lower_is_better:
                        if m1 < m2:
                            better, worse = a1, a2
                        else:
                            better, worse = a2, a1
                    else:
                        if m1 > m2:
                            better, worse = a1, a2
                        else:
                            better, worse = a2, a1

                    if p_two < 0.01:
                        marker = '\u2021'
                        significance_markers[metric][problem][better] += marker
                        win_counts[better] += 1
                    elif p_two < 0.05:
                        marker = '\u2020'
                        significance_markers[metric][problem][better] += marker
                        win_counts[better] += 1

    return significance_markers, win_counts


def _get_problem_default_nvar(problem_name):
    """获取问题的默认变量数"""
    problem = _create_problem_by_name(problem_name)
    if problem:
        return problem.n_var
    return 0


def _get_problem_default_nobj(problem_name):
    """获取问题的默认目标数"""
    problem = _create_problem_by_name(problem_name)
    if problem:
        return problem.n_obj
    return 0


def _save_benchmark_to_history():
    """将当前评测结果保存到历史列表（最多保留10次）"""
    if (st.session_state.benchmark_results is None
            or st.session_state.benchmark_results['end_time'] is None):
        return

    history_entry = {
        'id': datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
        'timestamp': datetime.now(),
        'config': json.loads(json.dumps(st.session_state.benchmark_results['config'],
                                        default=str)),
        'results': st.session_state.benchmark_results,
    }

    st.session_state.benchmark_history.insert(0, history_entry)

    if len(st.session_state.benchmark_history) > 10:
        st.session_state.benchmark_history = st.session_state.benchmark_history[:10]


def _compute_ranking_changes(current_ranking, history_ranking):
    """计算两次评测之间的综合排名变动

    Args:
        current_ranking: [(algo, score), ...] 当前综合排名
        history_ranking: [(algo, score), ...] 历史综合排名

    Returns:
        list of dict: 每个算法的排名变动信息
    """
    cur_pos = {algo: rank + 1 for rank, (algo, _) in enumerate(current_ranking)}
    hist_pos = {algo: rank + 1 for rank, (algo, _) in enumerate(history_ranking)}

    all_algos = set(cur_pos.keys()) | set(hist_pos.keys())
    changes = []

    for algo in sorted(all_algos):
        cur = cur_pos.get(algo, '-')
        hist = hist_pos.get(algo, '-')
        if isinstance(cur, int) and isinstance(hist, int):
            delta = hist - cur
            if delta > 0:
                direction = f"上升{delta}位"
            elif delta < 0:
                direction = f"下降{abs(delta)}位"
            else:
                direction = "保持不变"
            change_text = f"{algo}: {hist}→{cur}, {direction}"
        else:
            change_text = f"{algo}: 仅{'当前' if cur != '-' else '历史'}评测存在"
        changes.append({
            'algo': algo,
            'from': hist,
            'to': cur,
            'delta': delta if (isinstance(cur, int) and isinstance(hist, int)) else None,
            'text': change_text
        })

    return changes


def _export_benchmark_csv(benchmark_results, selected_metrics):
    """导出Benchmark结果为CSV"""
    rows = []

    for algo_name, problems in benchmark_results['results'].items():
        for prob_name, data in problems.items():
            for repeat_data in data['repeats']:
                row = {
                    '算法': algo_name,
                    '问题': prob_name,
                    '重复次数': repeat_data['run_idx'] + 1,
                    '随机种子': repeat_data['seed'],
                    '运行时间(秒)': repeat_data.get('runtime', np.nan),
                    '运行状态': '成功' if repeat_data['error'] is None else '失败',
                    '错误信息': repeat_data['error'] if repeat_data['error'] else ''
                }
                for metric in selected_metrics:
                    row[metric] = repeat_data['metrics'].get(metric, np.nan)
                rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8-sig')


def benchmark_tab():
    """批量自动化Benchmark评测标签页"""
    st.header("🏆 批量自动化Benchmark评测")
    st.caption("配置算法和问题集合，自动运行所有组合并生成结构化评测报告")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        _render_benchmark_config()

    with right_col:
        _render_benchmark_report()


def _render_benchmark_config():
    """渲染左侧评测配置区"""
    st.subheader("⚙️ 评测配置")

    all_problems_flat = []
    for series, problems in PROBLEM_MAP.items():
        all_problems_flat.extend(problems.keys())

    st.markdown("**1. 算法集合（选择2~5个）**")
    selected_algorithms = st.multiselect(
        "选择待评测算法",
        list(ALGORITHM_MAP.keys()),
        default=[list(ALGORITHM_MAP.keys())[0], list(ALGORITHM_MAP.keys())[1]] if len(ALGORITHM_MAP) >= 2 else [],
        key="benchmark_algos"
    )

    n_algos = len(selected_algorithms)
    if n_algos < 2:
        st.warning("⚠️ 请至少选择2个算法")
    elif n_algos > 5:
        st.warning("⚠️ 最多只能选择5个算法")

    st.divider()
    st.markdown("**2. 问题集合（选择2~10个，可跨系列）**")

    problem_options = []
    for series, problems in PROBLEM_MAP.items():
        for prob in problems.keys():
            problem_options.append(f"{series[:-2]} - {prob}")

    selected_problem_labels = st.multiselect(
        "选择测试函数",
        problem_options,
        default=["ZDT - ZDT1", "DTLZ - DTLZ2"],
        key="benchmark_problems"
    )

    selected_problems = [label.split(" - ")[1] for label in selected_problem_labels]
    n_problems = len(selected_problems)

    if n_problems < 2:
        st.warning("⚠️ 请至少选择2个测试函数")
    elif n_problems > 10:
        st.warning("⚠️ 最多只能选择10个测试函数")

    if n_problems >= 1:
        has_dtlz_3obj = False
        for pname in selected_problems:
            if pname.startswith('DTLZ'):
                nobj = _get_problem_default_nobj(pname)
                if nobj >= 3:
                    has_dtlz_3obj = True
                    break
        if has_dtlz_3obj and 'NSGA-II' in selected_algorithms:
            st.markdown(
                '<div style="background-color: #FFF3E0; padding: 10px; border-left: 5px solid #FF9800; '
                'border-radius: 4px; margin: 8px 0;">'
                '<span style="color: #E65100;">⚠️ '
                '<b>NSGA-II</b>不适合3目标以上问题，建议改用NSGA-III或MOEA/D</span>'
                '</div>',
                unsafe_allow_html=True
            )

    if n_problems >= 2:
        nvars = [_get_problem_default_nvar(p) for p in selected_problems]
        nvars_valid = [v for v in nvars if v > 0]
        if len(nvars_valid) >= 2:
            max_var, min_var = max(nvars_valid), min(nvars_valid)
            if min_var > 0 and (max_var / min_var) > 5:
                st.markdown(
                    '<div style="background-color: #FFF3E0; padding: 10px; border-left: 5px solid #FF9800; '
                    'border-radius: 4px; margin: 8px 0;">'
                    '<span style="color: #E65100;">💡 '
                    f'所选问题的变量维度差异较大({min_var}~{max_var})，'
                    '建议统一变量数或分批评测</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

    st.divider()
    st.markdown("**3. 统一运行参数**")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        n_gen = st.number_input("最大代数", min_value=10, max_value=5000, value=200, step=10, key="benchmark_n_gen")
    with col_p2:
        pop_size = st.number_input("种群大小", min_value=10, max_value=1000, value=100, step=10, key="benchmark_pop_size")

    col_p3, col_p4 = st.columns(2)
    with col_p3:
        n_repeats = st.number_input("重复运行次数", min_value=1, max_value=10, value=3, step=1, key="benchmark_n_repeats")
    with col_p4:
        seed_start = st.number_input("随机种子起始值", min_value=0, value=42, step=1, key="benchmark_seed_start")

    if n_repeats < 1 or n_repeats > 10:
        st.error("❌ 重复次数必须在1~10之间")

    st.divider()
    st.markdown("**4. 评测指标选择（至少选1个）**")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        select_igd = st.checkbox("IGD", value=True, key="benchmark_metric_igd")
    with col_m2:
        select_hv = st.checkbox("HV", value=True, key="benchmark_metric_hv")
    with col_m3:
        select_spacing = st.checkbox("Spacing", value=True, key="benchmark_metric_spacing")

    selected_metrics = []
    if select_igd:
        selected_metrics.append('IGD')
    if select_hv:
        selected_metrics.append('HV')
    if select_spacing:
        selected_metrics.append('Spacing')

    if len(selected_metrics) == 0:
        st.warning("⚠️ 请至少选择1个评测指标")

    st.divider()

    total_combinations = n_algos * n_problems * n_repeats
    st.info(f"📊 总组合数: {n_algos}算法 × {n_problems}问题 × {n_repeats}重复 = **{total_combinations}** 个组合")

    can_start = (
        2 <= n_algos <= 5 and
        2 <= n_problems <= 10 and
        1 <= n_repeats <= 10 and
        len(selected_metrics) >= 1 and
        not st.session_state.benchmark_running
    )

    col_run, col_stop = st.columns(2)
    with col_run:
        start_btn = st.button(
            "🚀 启动批量评测",
            type="primary",
            use_container_width=True,
            disabled=not can_start,
            key="btn_benchmark_start"
        )
    with col_stop:
        stop_btn = st.button(
            "⏹️ 终止",
            use_container_width=True,
            disabled=not st.session_state.benchmark_running,
            key="btn_benchmark_stop"
        )

    pending = st.session_state.benchmark_pending_start

    if pending is not None:
        st.warning(
            f"⚠️ 评测规模较大({pending['total_combinations']}个组合)，"
            "预计耗时较长，是否继续？"
        )
        col_y, col_n = st.columns(2)
        with col_y:
            confirm_btn = st.button(
                "✅ 确认继续",
                type="primary",
                use_container_width=True,
                key="btn_benchmark_confirm"
            )
        with col_n:
            cancel_btn = st.button(
                "❌ 取消",
                use_container_width=True,
                key="btn_benchmark_cancel"
            )
        if confirm_btn:
            st.session_state.benchmark_pending_start = None
            _start_benchmark(
                pending['algorithms'], pending['problems'], pending['metrics'],
                pending['n_gen'], pending['pop_size'], pending['n_repeats'],
                pending['seed_start'], pending['total_combinations']
            )
        if cancel_btn:
            st.session_state.benchmark_pending_start = None
            st.rerun()

    if start_btn and can_start and pending is None:
        if total_combinations > 100:
            st.session_state.benchmark_pending_start = {
                'algorithms': selected_algorithms,
                'problems': selected_problems,
                'metrics': selected_metrics,
                'n_gen': n_gen,
                'pop_size': pop_size,
                'n_repeats': n_repeats,
                'seed_start': seed_start,
                'total_combinations': total_combinations
            }
            st.rerun()
        else:
            _start_benchmark(selected_algorithms, selected_problems, selected_metrics,
                            n_gen, pop_size, n_repeats, seed_start, total_combinations)

    if stop_btn and st.session_state.benchmark_running:
        st.session_state.benchmark_running = False
        st.warning("⚠️ 评测已终止")


def _start_benchmark(selected_algorithms, selected_problems, selected_metrics,
                     n_gen, pop_size, n_repeats, seed_start, total_combinations):
    """启动批量评测"""
    config = {
        'algorithms': selected_algorithms,
        'problems': selected_problems,
        'metrics': selected_metrics,
        'n_gen': n_gen,
        'pop_size': pop_size,
        'n_repeats': n_repeats,
        'seed_start': seed_start,
        'total_combinations': total_combinations
    }

    st.session_state.benchmark_config = config
    st.session_state.benchmark_running = True
    st.session_state.benchmark_progress = 0

    results = {}
    for algo in selected_algorithms:
        results[algo] = {}
        for prob in selected_problems:
            results[algo][prob] = {
                'repeats': [],
                'stats': {},
                'has_error': False
            }

    st.session_state.benchmark_results = {
        'config': config,
        'results': results,
        'completed': 0,
        'total': total_combinations,
        'current_algo': None,
        'current_problem': None,
        'current_repeat': None,
        'start_time': datetime.now(),
        'end_time': None
    }

    _run_benchmark_loop(selected_algorithms, selected_problems, selected_metrics,
                       n_gen, pop_size, n_repeats, seed_start)


def _run_benchmark_loop(selected_algorithms, selected_problems, selected_metrics,
                        n_gen, pop_size, n_repeats, seed_start):
    """运行评测主循环（在单次脚本执行内完成，避免st.rerun()中断执行）"""
    try:
        completed = 0
        total = st.session_state.benchmark_results['total']
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        for problem_idx, problem_name in enumerate(selected_problems):
            if not st.session_state.benchmark_running:
                break

            problem = _create_problem_by_name(problem_name)
            if problem is None:
                for algo_name in selected_algorithms:
                    for run_idx in range(n_repeats):
                        st.session_state.benchmark_results['results'][algo_name][problem_name]['repeats'].append({
                            'run_idx': run_idx,
                            'seed': seed_start + problem_idx * 100 + run_idx,
                            'metrics': {},
                            'final_population': None,
                            'runtime': 0,
                            'error': f"无法创建问题实例: {problem_name}"
                        })
                        st.session_state.benchmark_results['results'][algo_name][problem_name]['has_error'] = True
                        completed += 1
                        st.session_state.benchmark_results['completed'] = completed
                continue

            for algo_idx, algo_name in enumerate(selected_algorithms):
                if not st.session_state.benchmark_running:
                    break

                st.session_state.benchmark_results['current_algo'] = algo_name
                st.session_state.benchmark_results['current_problem'] = problem_name

                repeats_data = []

                for run_idx in range(n_repeats):
                    if not st.session_state.benchmark_running:
                        break

                    st.session_state.benchmark_results['current_repeat'] = run_idx + 1

                    seed = seed_start + problem_idx * 100 + algo_idx * 10 + run_idx

                    repeat_result = {
                        'run_idx': run_idx,
                        'seed': seed,
                        'metrics': {},
                        'final_population': None,
                        'runtime': 0,
                        'error': None
                    }

                    try:
                        algorithm = _create_algorithm_by_name(algo_name, pop_size, n_gen, seed)

                        start_time = time.time()
                        result = algorithm.run(problem, verbose=False)
                        runtime = time.time() - start_time

                        approx_front = result.pareto_front
                        metrics = _compute_benchmark_metrics(approx_front, problem, selected_metrics)

                        repeat_result['metrics'] = metrics
                        repeat_result['final_population'] = approx_front
                        repeat_result['runtime'] = runtime

                        record = ExperimentRecord(
                            algorithm_name=algo_name,
                            problem_name=problem.name,
                            params=algorithm.get_params(),
                            metrics=metrics,
                            runtime=runtime,
                            result=result
                        )
                        st.session_state.experiment_history.add_record(record)

                    except Exception as e:
                        repeat_result['error'] = str(e)
                        st.session_state.benchmark_results['results'][algo_name][problem_name]['has_error'] = True

                    repeats_data.append(repeat_result)
                    completed += 1
                    st.session_state.benchmark_progress = completed / total
                    st.session_state.benchmark_results['completed'] = completed

                    st.session_state.benchmark_results['results'][algo_name][problem_name]['repeats'] = repeats_data
                    st.session_state.benchmark_results['results'][algo_name][problem_name]['stats'] = \
                        _compute_benchmark_stats(repeats_data, selected_metrics)

                    progress_placeholder.progress(completed / total)
                    status_placeholder.info(
                        f"📊 进度: {completed}/{total} 个组合 | "
                        f"当前: {algo_name} + {problem_name} (重复 {run_idx + 1}/{n_repeats})"
                    )

        st.session_state.benchmark_running = False
        st.session_state.benchmark_results['current_algo'] = None
        st.session_state.benchmark_results['current_problem'] = None
        st.session_state.benchmark_results['current_repeat'] = None
        st.session_state.benchmark_results['end_time'] = datetime.now()
        _save_benchmark_to_history()
        progress_placeholder.empty()
        status_placeholder.empty()
        st.rerun()

    except Exception as e:
        st.session_state.benchmark_running = False
        st.error(f"❌ 评测过程中发生错误: {e}")


def _render_benchmark_report():
    """渲染右侧报告展示区（集成历史记录、显著性检验、权重排名）"""

    st.markdown("#### 📜 历史评测")
    history = st.session_state.benchmark_history
    history_options = ["-- 当前评测结果 --"]
    if history:
        for h in history:
            ts = h['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            cfg = h.get('config', {})
            algos = ",".join(cfg.get('algorithms', []))
            probs = ",".join(cfg.get('problems', []))
            history_options.append(f"{ts} | {algos} | {probs}")

    selected_history_idx = st.selectbox(
        "选择历史评测记录（切换查看/对比）",
        range(len(history_options)),
        format_func=lambda i: history_options[i],
        index=0,
        key="benchmark_history_selector"
    )

    display_benchmark_results = st.session_state.benchmark_results
    display_config = (display_benchmark_results['config']
                      if display_benchmark_results else None)
    is_history_view = selected_history_idx > 0

    if is_history_view:
        hist_entry = history[selected_history_idx - 1]
        display_benchmark_results = hist_entry['results']
        display_config = display_benchmark_results['config']
        st.info(f"📜 正在查看历史评测: {history_options[selected_history_idx]}")
        if (st.session_state.benchmark_results is not None
                and st.session_state.benchmark_results['completed'] > 0):
            try:
                cur_cfg = st.session_state.benchmark_results['config']
                cur_metrics = cur_cfg['metrics']
                cur_rankings, cur_overall, _ = _compute_benchmark_rankings(
                    st.session_state.benchmark_results, cur_metrics
                )
                hist_metrics = display_config['metrics']
                hist_rankings, hist_overall, _ = _compute_benchmark_rankings(
                    display_benchmark_results, hist_metrics
                )
                if cur_metrics == hist_metrics:
                    changes = _compute_ranking_changes(cur_overall, hist_overall)
                    st.markdown("**🔀 对比变化（历史→当前）综合排名变动：**")
                    change_lines = []
                    for c in changes:
                        if c['delta'] is not None and c['delta'] > 0:
                            emoji = "🟢"
                        elif c['delta'] is not None and c['delta'] < 0:
                            emoji = "🔴"
                        else:
                            emoji = "⚪"
                        change_lines.append(f"{emoji} {c['text']}")
                    st.markdown("  \n".join(change_lines))
            except Exception:
                pass

    is_running_now = (not is_history_view) and st.session_state.benchmark_running

    if display_benchmark_results is None and not is_running_now:
        st.info("👈 请在左侧配置评测参数后点击\"启动批量评测\"")
        return

    if is_running_now:
        display_benchmark_results = st.session_state.benchmark_results
        if display_benchmark_results is None:
            st.info("⏳ 正在初始化评测...请稍候")
            return
        display_config = display_benchmark_results['config']

    if display_config is None:
        return
    selected_metrics = display_config['metrics']

    if is_running_now:
        st.subheader("🔄 评测进行中...")
        st.info(
            "评测正在后台执行，请等待完成。完成后将自动刷新页面展示完整结果。"
        )
        progress = st.session_state.benchmark_progress
        st.progress(progress)
        completed = display_benchmark_results.get('completed', 0)
        total = display_benchmark_results.get('total', 0)
        current_algo = display_benchmark_results.get('current_algo')
        current_problem = display_benchmark_results.get('current_problem')
        current_repeat = display_benchmark_results.get('current_repeat')
        n_repeats = display_config.get('n_repeats', '?')
        st.caption(
            f"📊 进度: {completed}/{total} 个组合 "
            f"({progress * 100:.1f}%) | "
            f"当前: {current_algo} + {current_problem} "
            f"(重复 {current_repeat}/{n_repeats})"
        )
        if completed == 0:
            return

    if display_benchmark_results['completed'] > 0:
        st.subheader("📈 评测结果")

        with st.expander("⚖️ 指标权重配置（调整后排名实时刷新）", expanded=False):
            st.caption("每个(问题,指标)的排名分将乘以对应指标的权重后求和，得到综合排名分数")
            metric_weights = {}
            w_cols = st.columns(len(selected_metrics))
            for i, metric in enumerate(selected_metrics):
                default_w = st.session_state.benchmark_metric_weights.get(metric, 1.0)
                with w_cols[i]:
                    w = st.slider(
                        f"{metric} 权重",
                        min_value=0.1,
                        max_value=5.0,
                        value=float(default_w),
                        step=0.1,
                        key=f"weight_slider_{metric}_{selected_history_idx}"
                    )
                    metric_weights[metric] = w
                    st.caption(f"权重值: {w:.1f}")
            st.session_state.benchmark_metric_weights.update(metric_weights)

        rankings, overall_ranking, overall_scores = _compute_benchmark_rankings(
            display_benchmark_results, selected_metrics, metric_weights
        )

        significance_markers, win_counts = _compute_benchmark_significance(
            display_benchmark_results, selected_metrics
        )

        for metric in selected_metrics:
            st.markdown(f"### {metric} 汇总表")
            metric_optim = "(越小越好)" if metric in ['IGD', 'Spacing'] else "(越大越好)"
            st.caption(
                f"{metric} {metric_optim} | "
                "显著性标记: ‡ p<0.01, † p<0.05 (重复次数<3不标注)"
            )

            algo_names = display_config['algorithms']
            problem_names = display_config['problems']

            table_data = []
            best_values = {}

            for problem in problem_names:
                valid_values = []
                for algo in algo_names:
                    stats = display_benchmark_results['results'][algo][problem]['stats']
                    if metric in stats and stats[metric]['count'] > 0:
                        valid_values.append(stats[metric]['mean'])

                if valid_values:
                    if metric in ['IGD', 'Spacing']:
                        best_values[problem] = min(valid_values)
                    else:
                        best_values[problem] = max(valid_values)

            for algo in algo_names:
                row = {'算法': algo}
                for problem in problem_names:
                    stats = display_benchmark_results['results'][algo][problem]['stats']
                    marker = significance_markers.get(metric, {}).get(problem, {}).get(algo, '')
                    if metric in stats and stats[metric]['count'] > 0:
                        cell_value = _format_mean_std(stats[metric]['mean'], stats[metric]['std'])
                    else:
                        cell_value = "N/A"
                    if marker:
                        cell_value = f"{cell_value}<sup style='color:#d32f2f; font-weight:bold;'>{marker}</sup>"
                    row[problem] = cell_value
                table_data.append(row)

            win_row = {'算法': '<b>显著胜出次数</b>'}
            for problem in problem_names:
                win_counts_per_problem = {}
                for algo in algo_names:
                    marker = significance_markers.get(metric, {}).get(problem, {}).get(algo, '')
                    win_counts_per_problem[algo] = len([c for c in marker])
                cell_parts = []
                for algo in algo_names:
                    wc = win_counts_per_problem[algo]
                    if wc > 0:
                        cell_parts.append(f"{algo}:{wc}")
                win_row[problem] = "<br>".join(cell_parts) if cell_parts else "-"
            table_data.append(win_row)

            df = pd.DataFrame(table_data)

            def highlight_best(row):
                styles = [''] * len(row)
                if row['算法'] == '<b>显著胜出次数</b>':
                    return ['background-color: #FFF8E1; color: black; font-weight: bold;'] * len(row)
                for i, problem in enumerate(problem_names):
                    col_idx = i + 1
                    stats = display_benchmark_results['results'][row['算法']][problem]['stats']
                    if metric in stats and stats[metric]['count'] > 0:
                        if abs(stats[metric]['mean'] - best_values.get(problem, float('inf'))) < 1e-10:
                            styles[col_idx] = 'background-color: #90EE90; color: black;'
                return styles

            try:
                styled_df = df.style.apply(highlight_best, axis=1)
                html_table = styled_df.to_html(escape=False, index=False)
                st.markdown(html_table, unsafe_allow_html=True)
            except Exception:
                st.dataframe(df, use_container_width=True, hide_index=True,
                             key=f"df_benchmark_{metric}_{selected_history_idx}")

        st.markdown("### 🏆 综合排名")
        ranking_data = []
        for rank, (algo, score) in enumerate(overall_ranking):
            ranking_data.append({
                '排名': rank + 1,
                '算法': algo,
                '总分': score,
                '显著胜出次数': win_counts.get(algo, 0)
            })
        ranking_df = pd.DataFrame(ranking_data)
        st.dataframe(
            ranking_df.style.format({'总分': '{:.1f}', '显著胜出次数': '{:d}'}),
            use_container_width=True,
            hide_index=True,
            key=f"df_benchmark_ranking_{selected_history_idx}"
        )

        st.divider()
        st.markdown("### 📋 详细结果")

        for algo in display_config['algorithms']:
            for problem in display_config['problems']:
                data = display_benchmark_results['results'][algo][problem]
                has_error = data['has_error']
                error_tag = " ❌" if has_error else ""

                with st.expander(f"📊 {algo} + {problem}{error_tag}", expanded=False):
                    if has_error:
                        failed_runs = [r for r in data['repeats'] if r['error'] is not None]
                        if failed_runs:
                            st.error(f"❌ 存在运行错误: {failed_runs[0]['error']}")

                    detail_rows = []
                    for repeat_data in data['repeats']:
                        row = {
                            '重复': repeat_data['run_idx'] + 1,
                            '种子': repeat_data['seed'],
                            '运行时间(s)': f"{repeat_data['runtime']:.2f}",
                            '状态': '✅ 成功' if repeat_data['error'] is None else f'❌ {repeat_data["error"]}'
                        }
                        for metric in selected_metrics:
                            val = repeat_data['metrics'].get(metric, np.nan)
                            row[metric] = f"{val:.4f}" if not np.isnan(val) else "N/A"
                        detail_rows.append(row)

                    if detail_rows:
                        detail_df = pd.DataFrame(detail_rows)
                        st.dataframe(detail_df, use_container_width=True, hide_index=True,
                                     key=f"df_benchmark_detail_{algo}_{problem}_{selected_history_idx}")

                    if data['stats']:
                        st.markdown("**统计值:**")
                        stats_cols = st.columns(len(selected_metrics))
                        for i, metric in enumerate(selected_metrics):
                            if metric in data['stats'] and data['stats'][metric]['count'] > 0:
                                s = data['stats'][metric]
                                with stats_cols[i]:
                                    st.metric(
                                        f"{metric}",
                                        f"{s['mean']:.4f}",
                                        f"±{s['std']:.4f}"
                                    )

                    last_pop = None
                    for repeat_data in reversed(data['repeats']):
                        if repeat_data['final_population'] is not None:
                            last_pop = repeat_data['final_population']
                            break

                    if last_pop is not None:
                        st.markdown("**目标空间散点图（最后一次重复结果）:**")
                        problem_obj = _create_problem_by_name(problem)
                        true_front = (problem_obj.pareto_front()
                                      if problem_obj and problem_obj.pareto_front() is not None
                                      else None)

                        if last_pop.shape[1] == 2:
                            fig = plot_pareto_front_2d(
                                {'近似前沿': last_pop},
                                true_front=true_front,
                                title=f"{algo} + {problem} - 目标空间"
                            )
                            st.pyplot(fig, use_container_width=True)
                        elif last_pop.shape[1] == 3:
                            fig = plot_pareto_front_3d(
                                {'近似前沿': last_pop},
                                true_front=true_front,
                                title=f"{algo} + {problem} - 目标空间"
                            )
                            st.pyplot(fig, use_container_width=True)
                        else:
                            fig = plot_parallel_coordinates(
                                {'近似前沿': last_pop},
                                title=f"{algo} + {problem} - 平行坐标图"
                            )
                            st.pyplot(fig, use_container_width=True)

        st.divider()
        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            csv_data = _export_benchmark_csv(display_benchmark_results, selected_metrics)
            fname = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                "📥 导出完整报告",
                csv_data,
                fname,
                "text/csv",
                use_container_width=True,
                key=f"btn_benchmark_export_{selected_history_idx}"
            )
        with col_dl2:
            st.caption("CSV文件包含所有组合所有重复的原始指标数据、统计值和运行信息")


def main():
    """主函数"""
    init_session_state()

    st.title("🎯 多目标进化优化实验与帕累托前沿分析平台")
    st.caption("支持 ZDT/DTLZ/WFG 系列测试函数 · NSGA-II/NSGA-III/MOEA/D/SPEA2/SMS-EMOA 算法")

    problem = sidebar_problem_config()
    sidebar_algorithm_config()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📚 问题定义",
        "▶️ 运行优化",
        "📊 结果可视化",
        "🔬 对比分析",
        "🎯 决策支持",
        "📐 参数灵敏度分析",
        "📈 算法收敛性监控",
        "📋 实验记录",
        "🏆 批量Benchmark评测"
    ])

    with tab1:
        problem_info_tab(problem)

    with tab2:
        run_optimization_tab()

    with tab3:
        st.header("📊 结果可视化")
        if st.session_state.current_results:
            display_results(st.session_state.current_problem)
        else:
            st.info("请先运行优化实验")

    with tab4:
        comparison_tab()

    with tab5:
        decision_support_tab()

    with tab6:
        sensitivity_analysis_tab()

    with tab7:
        convergence_monitoring_tab()

    with tab8:
        history_tab()

    with tab9:
        benchmark_tab()

    st.divider()
    st.caption("帕累托前沿分析平台 v1.0 | 基于 Streamlit + NumPy + Matplotlib")


if __name__ == "__main__":
    main()
