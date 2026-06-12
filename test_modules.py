"""测试脚本 - 验证模块导入和基本功能"""
import sys
import traceback

def test_imports():
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)

    modules = [
        ("pareto_moea", "主包"),
        ("pareto_moea.problems", "问题定义模块"),
        ("pareto_moea.utils", "工具模块"),
        ("pareto_moea.algorithms", "算法模块"),
        ("pareto_moea.metrics", "性能指标模块"),
        ("pareto_moea.visualization", "可视化模块"),
        ("pareto_moea.decision_making", "决策支持模块"),
        ("pareto_moea.experiments", "实验记录模块"),
    ]

    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description} ({module_name}) - OK")
        except Exception as e:
            print(f"✗ {description} ({module_name}) - 失败: {e}")
            traceback.print_exc()
            return False

    print("\n所有模块导入成功！\n")
    return True


def test_problems():
    print("=" * 60)
    print("测试问题定义模块...")
    print("=" * 60)

    from pareto_moea.problems import ZDT1, ZDT2, DTLZ2, WFG4, CustomProblem
    import numpy as np

    # 测试ZDT1
    zdt1 = ZDT1(n_var=10)
    x = np.random.rand(10)
    f = zdt1.evaluate(x)
    print(f"✓ ZDT1: n_var={zdt1.n_var}, n_obj={zdt1.n_obj}, f={f}")

    pf = zdt1.pareto_front(100)
    print(f"  帕累托前沿: {pf.shape}")

    # 测试DTLZ2
    dtlz2 = DTLZ2(n_obj=3, k=5)
    x = np.random.rand(dtlz2.n_var)
    f = dtlz2.evaluate(x)
    print(f"✓ DTLZ2: n_var={dtlz2.n_var}, n_obj={dtlz2.n_obj}, f={f}")

    # 测试自定义问题
    custom = CustomProblem(
        objective_func=lambda x: np.array([sum(x**2), sum((x-1)**2)]),
        n_var=5,
        n_obj=2,
        xl=np.zeros(5),
        xu=np.ones(5)
    )
    x = np.random.rand(5)
    f = custom.evaluate(x)
    print(f"✓ 自定义问题: f={f}")

    # 批量评估
    pop = np.random.rand(20, 10)
    f_batch = zdt1.evaluate(pop)
    print(f"✓ 批量评估: {f_batch.shape}")

    print()
    return True


def test_utils():
    print("=" * 60)
    print("测试工具模块...")
    print("=" * 60)

    from pareto_moea.utils import (
        fast_non_dominated_sort,
        crowding_distance,
        pareto_front,
        uniform_reference_points,
        sbx_crossover,
        polynomial_mutation,
        penalty_function,
        feasibility_rule
    )
    import numpy as np

    # 测试非支配排序
    objectives = np.random.rand(50, 2)
    ranks = fast_non_dominated_sort(objectives)
    print(f"✓ 快速非支配排序: {len(np.unique(ranks))} 个前沿")

    # 测试拥挤距离
    front = pareto_front(objectives)
    cd = crowding_distance(front)
    print(f"✓ 拥挤距离计算: {len(cd)} 个点")

    # 测试参考点生成
    ref_points = uniform_reference_points(3, 10)
    print(f"✓ 参考点生成: {ref_points.shape}")

    # 测试遗传算子
    xl = np.zeros(10)
    xu = np.ones(10)
    p1 = np.random.rand(10)
    p2 = np.random.rand(10)
    c1, c2 = sbx_crossover(p1, p2, xl, xu, 20.0, 0.9)
    print(f"✓ SBX交叉")

    mutant = polynomial_mutation(p1, xl, xu, 20.0, 0.1)
    print(f"✓ 多项式变异")

    # 测试约束处理
    obj = np.random.rand(10, 2)
    cv = np.random.rand(10, 1) * 0.1
    penalty = penalty_function(obj, cv, 10.0)
    print(f"✓ 罚函数法")

    print()
    return True


def test_algorithms():
    print("=" * 60)
    print("测试算法模块...")
    print("=" * 60)

    from pareto_moea.algorithms import NSGA2, NSGA3, MOEAD, SPEA2, SMSEMOA
    from pareto_moea.problems import ZDT1
    import numpy as np

    problem = ZDT1(n_var=10)

    algorithms = [
        ("NSGA-II", NSGA2(pop_size=50, n_gen=10, seed=42)),
        ("NSGA-III", NSGA3(pop_size=50, n_gen=10, n_divisions=6, seed=42)),
        ("MOEA/D", MOEAD(pop_size=50, n_gen=10, n_weights=50, neighbor_size=10, seed=42)),
        ("SPEA2", SPEA2(pop_size=50, n_gen=10, archive_size=50, seed=42)),
        ("SMS-EMOA", SMSEMOA(pop_size=50, n_gen=10, seed=42)),
    ]

    for name, algo in algorithms:
        try:
            result = algo.run(problem, verbose=False)
            print(f"✓ {name}: pf={result.pareto_front.shape}, runtime={result.runtime:.2f}s")
        except Exception as e:
            print(f"✗ {name}: 失败 - {e}")
            traceback.print_exc()

    print()
    return True


def test_metrics():
    print("=" * 60)
    print("测试性能指标模块...")
    print("=" * 60)

    from pareto_moea.metrics import gd, igd, hv, spacing, spread, pairwise_wilcoxon
    from pareto_moea.problems import ZDT1
    import numpy as np

    zdt1 = ZDT1(n_var=10)
    true_front = zdt1.pareto_front(500)

    approx = true_front + np.random.randn(*true_front.shape) * 0.01

    print(f"✓ GD: {gd(approx, true_front):.6f}")
    print(f"✓ IGD: {igd(approx, true_front):.6f}")
    print(f"✓ Spacing: {spacing(approx):.6f}")
    print(f"✓ Spread: {spread(approx, true_front):.6f}")

    ref_point = np.array([1.1, 1.1])
    print(f"✓ HV: {hv(approx, ref_point):.6f}")

    # 测试统计检验
    data = {
        'Algo1': np.random.rand(20) * 0.1,
        'Algo2': np.random.rand(20) * 0.2,
    }
    p_matrix, labels = pairwise_wilcoxon(data)
    print(f"✓ Wilcoxon检验: p={p_matrix[0, 1]:.4f}")

    print()
    return True


def test_visualization():
    print("=" * 60)
    print("测试可视化模块...")
    print("=" * 60)

    from pareto_moea.visualization import (
        plot_pareto_front_2d,
        plot_pareto_front_3d,
        plot_parallel_coordinates,
        plot_convergence,
        plot_boxplot
    )
    from pareto_moea.problems import ZDT1, DTLZ2
    import numpy as np

    # 2D图
    zdt1 = ZDT1()
    true_front = zdt1.pareto_front(100)
    approx = true_front + np.random.randn(*true_front.shape) * 0.02
    data = {'NSGA-II': approx}
    fig = plot_pareto_front_2d(data, true_front=true_front, title="Test 2D")
    print(f"✓ 2D散点图")

    # 3D图
    dtlz2 = DTLZ2(n_obj=3)
    true_3d = dtlz2.pareto_front(200)
    approx_3d = true_3d + np.random.randn(*true_3d.shape) * 0.02
    fig = plot_pareto_front_3d({'NSGA-II': approx_3d}, true_front=true_3d, title="Test 3D")
    print(f"✓ 3D散点图")

    # 平行坐标图
    fig = plot_parallel_coordinates({'Test': true_3d}, title="Parallel Coords")
    print(f"✓ 平行坐标图")

    # 收敛曲线
    conv_data = {
        'Algo1': np.logspace(0, -2, 50),
        'Algo2': np.logspace(0, -3, 50),
    }
    fig = plot_convergence(conv_data, title="Convergence")
    print(f"✓ 收敛曲线图")

    # 箱线图
    box_data = {
        'Algo1': np.random.rand(20) * 0.1,
        'Algo2': np.random.rand(20) * 0.2,
    }
    fig = plot_boxplot(box_data, title="Boxplot")
    print(f"✓ 箱线图")

    print()
    return True


def test_decision_making():
    print("=" * 60)
    print("测试决策支持模块...")
    print("=" * 60)

    from pareto_moea.decision_making import knee_point_detection, topsis, region_filter
    from pareto_moea.problems import ZDT1
    import numpy as np

    zdt1 = ZDT1()
    pf = zdt1.pareto_front(100)

    # 膝点检测
    idx, knee = knee_point_detection(pf, method='angle')
    print(f"✓ 膝点检测 (angle): idx={idx}, f={knee}")

    idx, knee = knee_point_detection(pf, method='distance')
    print(f"✓ 膝点检测 (distance): idx={idx}, f={knee}")

    # TOPSIS
    weights = np.array([0.5, 0.5])
    scores, ranks = topsis(pf, weights=weights, return_ranks=True)
    print(f"✓ TOPSIS排序: 最佳得分={scores[np.argmin(ranks)]:.4f}")

    # 区域筛选
    lower = np.array([0.2, 0.0])
    upper = np.array([0.8, 1.0])
    filtered = region_filter(pf, lower_bounds=lower, upper_bounds=upper)
    print(f"✓ 区域筛选: 筛选出 {len(filtered)} 个解")

    print()
    return True


def test_experiments():
    print("=" * 60)
    print("测试实验记录模块...")
    print("=" * 60)

    from pareto_moea.experiments import ExperimentHistory, ExperimentRecord
    import numpy as np

    history = ExperimentHistory()

    for i in range(5):
        history.add(
            algorithm_name='NSGA2',
            problem_name='ZDT1',
            params={'pop_size': 100, 'n_gen': 100},
            metrics={'HV': 0.8 + np.random.rand() * 0.1, 'IGD': 0.01 + np.random.rand() * 0.01},
            runtime=10.0 + np.random.rand() * 5,
            result=None
        )

    print(f"✓ 添加记录: {len(history)} 条记录")

    df = history.to_dataframe()
    print(f"✓ 转换DataFrame: {df.shape}")

    filtered = history.filter(algorithm_name='NSGA2')
    print(f"✓ 筛选记录: {len(filtered)} 条")

    summary = history.summary()
    print(f"✓ 统计汇总: {len(summary)} 行")

    print()
    return True


def main():
    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_problems()
    all_passed &= test_utils()
    all_passed &= test_algorithms()
    all_passed &= test_metrics()
    all_passed &= test_visualization()
    all_passed &= test_decision_making()
    all_passed &= test_experiments()

    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
