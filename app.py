"""多目标进化优化实验与帕累托前沿分析平台"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
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
    gd, igd, hv, spacing, spread,
    pairwise_wilcoxon, significance_level, compute_statistics
)

from pareto_moea.visualization import (
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_parallel_coordinates,
    plot_convergence,
    plot_boxplot,
    plot_generation_animation,
    plot_sensitivity_line
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
            st.dataframe(bounds_df.head(10), use_container_width=True)
            st.caption(f"... 共 {problem.n_var} 个变量")
        else:
            st.dataframe(bounds_df, use_container_width=True)

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
        start_button = st.button("🚀 开始优化", type="primary", use_container_width=True, disabled=st.session_state.running)

    with col3:
        st.markdown("### ")
        stop_button = st.button("⏹️ 终止", use_container_width=True, disabled=not st.session_state.running)

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
        st.dataframe(df, use_container_width=True)


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
            st.dataframe(p_df.style.format('{:.4f}'), use_container_width=True)

            st.caption("显著性水平: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著")

            sig_matrix = [[significance_level(p) for p in row] for row in p_matrix]
            sig_df = pd.DataFrame(sig_matrix, index=labels, columns=labels)
            st.dataframe(sig_df, use_container_width=True)

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

            st.dataframe(result_df, use_container_width=True)

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
                st.dataframe(df, use_container_width=True)

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
                disabled=st.session_state.sensitivity_running
            )
        with col_stop:
            stop_analysis = st.button(
                "⏹️ 终止",
                use_container_width=True,
                disabled=not st.session_state.sensitivity_running
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
                hide_index=True
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
                    use_container_width=True
                )
            with col_dl2:
                st.caption("CSV 文件包含各参数值下 IGD/GD/HV 的均值、标准差、最小值和最大值")


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

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 导出为 CSV", use_container_width=True):
            csv = filtered.to_csv(index=False)
            st.download_button(
                "下载 CSV 文件",
                csv,
                f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

    with col2:
        if st.button("🗑️ 清空所有记录", use_container_width=True, type="secondary"):
            history.clear()
            st.rerun()

    st.subheader("统计汇总")
    metric_cols = [c for c in df.columns if c.startswith('metric_')]
    if metric_cols:
        summary = history.summary()
        st.dataframe(summary, use_container_width=True)


def main():
    """主函数"""
    init_session_state()

    st.title("🎯 多目标进化优化实验与帕累托前沿分析平台")
    st.caption("支持 ZDT/DTLZ/WFG 系列测试函数 · NSGA-II/NSGA-III/MOEA/D/SPEA2/SMS-EMOA 算法")

    problem = sidebar_problem_config()
    sidebar_algorithm_config()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📚 问题定义",
        "▶️ 运行优化",
        "📊 结果可视化",
        "🔬 对比分析",
        "🎯 决策支持",
        "📐 参数灵敏度分析",
        "📋 实验记录"
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
        history_tab()

    st.divider()
    st.caption("帕累托前沿分析平台 v1.0 | 基于 Streamlit + NumPy + Matplotlib")


if __name__ == "__main__":
    main()
