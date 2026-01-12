"""
一键验证脚本 - 快速验证所有核心模块

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/modules/

This script verifies that all core modules are properly installed and functional:
- Gurobi solver interface
- SaboteurAgent error injection
- MDP environment (SolverDebugEnv)
- Agent implementations
- Evaluation framework (BenchmarkRunner)

Usage:
    python scripts/verify_installation.py
    python scripts/verify_installation.py --report outputs/report.md
    python scripts/verify_installation.py --verbose

Author: Ruicheng Ao
Created: 2026-01-11
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_failure(text: str):
    """Print failure message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def check_dependencies() -> Dict[str, Tuple[bool, str]]:
    """检查依赖包"""
    print_header("Checking Dependencies")

    dependencies = {}
    required_packages = {
        'gurobipy': 'Gurobi Python API',
        'gymnasium': 'Gymnasium RL framework',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
    }

    optional_packages = {
        'openai': 'OpenAI API (for LLMAgent)',
        'anthropic': 'Anthropic API (for LLMAgent)',
        'torch': 'PyTorch (for RL training)',
    }

    # Check required packages
    print("Required packages:")
    for package_name, description in required_packages.items():
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            dependencies[package_name] = (True, version)
            print_success(f"{package_name} ({description}) - v{version}")
        except ImportError as e:
            dependencies[package_name] = (False, str(e))
            print_failure(f"{package_name} ({description}) - NOT FOUND")

    # Check optional packages
    print("\nOptional packages:")
    for package_name, description in optional_packages.items():
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            dependencies[package_name] = (True, version)
            print_success(f"{package_name} ({description}) - v{version}")
        except ImportError:
            dependencies[package_name] = (False, "Not installed")
            print_warning(f"{package_name} ({description}) - Not installed (optional)")

    return dependencies


def verify_gurobi_solver() -> Tuple[bool, str]:
    """验证Gurobi求解器接口"""
    try:
        import gurobipy as gp
        from src.solvers import GurobiSolver

        # 创建简单模型
        m = gp.Model("verify_test")
        m.Params.OutputFlag = 0
        x = m.addVar(lb=0, ub=10, name="x")
        m.addConstr(x >= 5, name="c1")
        m.setObjective(x, gp.GRB.MAXIMIZE)
        m.update()

        # 测试求解
        solver = GurobiSolver.from_model(m)
        state = solver.solve()

        if state.status != "OPTIMAL":
            return False, f"Expected OPTIMAL, got {state.status}"

        if abs(state.objective - 10.0) > 1e-6:
            return False, f"Expected obj=10, got {state.objective}"

        # 测试IIS计算（在不可行模型上）
        m2 = gp.Model("infeasible_test")
        m2.Params.OutputFlag = 0
        y = m2.addVar(lb=0, ub=10, name="y")
        m2.addConstr(y >= 8, name="lower")
        m2.addConstr(y <= 5, name="upper")
        m2.setObjective(y, gp.GRB.MAXIMIZE)
        m2.update()

        solver2 = GurobiSolver.from_model(m2)
        state2 = solver2.solve()

        if state2.status != "INFEASIBLE":
            return False, f"Expected INFEASIBLE for conflicting model"

        iis = solver2.compute_iis()
        if iis.size == 0:
            return False, "IIS should not be empty for infeasible model"

        return True, f"Gurobi solver verified (optimal + IIS)"

    except Exception as e:
        return False, f"Error: {str(e)}"


def verify_saboteur_agent() -> Tuple[bool, str]:
    """验证错误注入"""
    try:
        import gurobipy as gp
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent

        # 创建可行模型
        m = gp.Model("feasible_for_sabotage")
        m.Params.OutputFlag = 0
        x = m.addVar(lb=0, ub=10, name="x")
        m.addConstr(x >= 0, name="c1")
        m.addConstr(x <= 10, name="c2")
        m.setObjective(x, gp.GRB.MAXIMIZE)
        m.update()

        solver = GurobiSolver.from_model(m)
        saboteur = SaboteurAgent(solver, seed=42)

        # 测试Type D注入（添加矛盾约束）
        result = saboteur.inject_type_d()

        if result.solver_status not in ["INFEASIBLE", "INF_OR_UNBD"]:
            return False, f"Expected INFEASIBLE after Type D, got {result.solver_status}"

        if not result.ground_truth_fix:
            return False, "Missing ground truth fix"

        if result.error_type.value != "D":
            return False, f"Expected error type D, got {result.error_type}"

        return True, f"Saboteur agent verified (Type D injection)"

    except Exception as e:
        return False, f"Error: {str(e)}"


def verify_mdp_environment() -> Tuple[bool, str]:
    """验证MDP环境"""
    try:
        import gurobipy as gp
        from src.solvers import GurobiSolver
        from src.environments import SolverDebugEnv, Action, ActionType

        # 创建不可行模型
        m = gp.Model("infeasible_env_test")
        m.Params.OutputFlag = 0
        x = m.addVar(lb=0, ub=10, name="x")
        m.addConstr(x >= 8, name="lower")
        m.addConstr(x <= 5, name="upper")
        m.setObjective(x, gp.GRB.MAXIMIZE)
        m.update()

        solver = GurobiSolver.from_model(m)
        env = SolverDebugEnv(solver, max_steps=10)

        # 测试reset
        state, info = env.reset()
        if not state.is_infeasible():
            return False, "Expected INFEASIBLE state after reset"

        # 测试step - GET_IIS
        action = Action(ActionType.GET_IIS)
        new_state, reward, terminated, truncated, info = env.step(action)

        if not info["action_result"]["success"]:
            return False, "GET_IIS action failed"

        if len(new_state.iis_constraints) == 0:
            return False, "No IIS constraints found after GET_IIS"

        # 测试step - DROP_CONSTRAINT
        target_constr = new_state.iis_constraints[0]
        action2 = Action(ActionType.DROP_CONSTRAINT, target=target_constr)
        new_state2, reward2, terminated2, truncated2, info2 = env.step(action2)

        if not info2["action_result"]["success"]:
            return False, "DROP_CONSTRAINT action failed"

        return True, f"MDP environment verified (reset + step)"

    except Exception as e:
        return False, f"Error: {str(e)}"


def verify_agents() -> Tuple[bool, str]:
    """验证Agent"""
    try:
        import gurobipy as gp
        from src.solvers import GurobiSolver
        from src.environments import SolverDebugEnv
        from src.agents import HeuristicAgent, RandomAgent, GreedyDropAgent

        # 创建环境
        m = gp.Model("agent_test")
        m.Params.OutputFlag = 0
        x = m.addVar(lb=0, ub=10, name="x")
        m.addConstr(x >= 8, name="lower")
        m.addConstr(x <= 5, name="upper")
        m.setObjective(x, gp.GRB.MAXIMIZE)
        m.update()

        solver = GurobiSolver.from_model(m)
        env = SolverDebugEnv(solver, max_steps=10)
        state, _ = env.reset()

        # 测试HeuristicAgent
        agent1 = HeuristicAgent()
        action1 = agent1.act(state)
        if action1 is None:
            return False, "HeuristicAgent returned None"

        # 测试RandomAgent
        agent2 = RandomAgent(seed=42)
        action2 = agent2.act(state)
        if action2 is None:
            return False, "RandomAgent returned None"

        # 测试GreedyDropAgent
        agent3 = GreedyDropAgent()
        action3 = agent3.act(state)
        if action3 is None:
            return False, "GreedyDropAgent returned None"

        return True, f"Agents verified (Heuristic, Random, Greedy)"

    except Exception as e:
        return False, f"Error: {str(e)}"


def verify_benchmark_runner() -> Tuple[bool, str]:
    """验证BenchmarkRunner"""
    try:
        import gurobipy as gp
        from src.solvers import GurobiSolver
        from src.environments import SolverDebugEnv
        from src.agents import HeuristicAgent
        from src.evaluation import (
            BenchmarkRunner,
            BenchmarkProblem,
            BenchmarkConfig,
            MetricsCalculator
        )

        # 创建问题
        m = gp.Model("benchmark_test")
        m.Params.OutputFlag = 0
        x = m.addVar(lb=0, ub=10, name="x")
        m.addConstr(x >= 8, name="lower")
        m.addConstr(x <= 5, name="upper")
        m.setObjective(x, gp.GRB.MAXIMIZE)
        m.update()

        solver = GurobiSolver.from_model(m)
        env = SolverDebugEnv(solver, max_steps=5)

        problems = [
            BenchmarkProblem(problem_id="test1", env=env)
        ]

        # 运行基准测试
        config = BenchmarkConfig(max_steps=5, n_episodes=1, verbose=False)
        runner = BenchmarkRunner(config=config)
        agent = HeuristicAgent()

        results = runner.run_benchmark(problems, agent)

        if len(results) != 1:
            return False, f"Expected 1 result, got {len(results)}"

        summary = runner.get_summary()
        if "recovery_rate" not in summary:
            return False, "Missing recovery_rate in summary"

        # 测试MetricsCalculator
        calc = MetricsCalculator()
        recovery_rate = calc.compute_recovery_rate(results)

        if not (0 <= recovery_rate <= 1):
            return False, f"Invalid recovery rate: {recovery_rate}"

        return True, f"BenchmarkRunner verified (runner + metrics)"

    except Exception as e:
        return False, f"Error: {str(e)}"


def run_verification(verbose: bool = False) -> Dict[str, Tuple[bool, str]]:
    """运行所有验证"""
    print_header("LLM-for-OR Installation Verification")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")

    # 检查依赖
    deps = check_dependencies()

    # 如果缺少必需依赖，提前退出
    required_deps = ['gurobipy', 'gymnasium', 'numpy']
    missing = [d for d in required_deps if not deps[d][0]]
    if missing:
        print_failure(f"\nMissing required dependencies: {', '.join(missing)}")
        print("Please install them before continuing.")
        return deps

    # 运行模块验证
    print_header("Module Verification")

    verifications = {
        "Gurobi Solver": verify_gurobi_solver,
        "Saboteur Agent": verify_saboteur_agent,
        "MDP Environment": verify_mdp_environment,
        "Agents": verify_agents,
        "Benchmark Runner": verify_benchmark_runner,
    }

    results = {}
    for name, verify_func in verifications.items():
        print(f"\n[{name}]")
        try:
            success, message = verify_func()
            results[name] = (success, message)
            if success:
                print_success(message)
            else:
                print_failure(message)
                if verbose:
                    traceback.print_exc()
        except Exception as e:
            results[name] = (False, f"Unexpected error: {str(e)}")
            print_failure(f"Unexpected error: {str(e)}")
            if verbose:
                traceback.print_exc()

    # 总结
    print_header("Verification Summary")
    total = len(results)
    passed = sum(1 for success, _ in results.values() if success)

    print(f"Total checks: {total}")
    print(f"Passed: {Colors.OKGREEN}{passed}{Colors.ENDC}")
    print(f"Failed: {Colors.FAIL}{total - passed}{Colors.ENDC}")
    print(f"Success rate: {passed/total*100:.1f}%\n")

    if passed == total:
        print_success("✓ All verifications passed!")
        print("\nYou are ready to:")
        print("  - Run experiments with demo/quickstart.py")
        print("  - Download MIPLIB datasets")
        print("  - Train LLM agents")
    else:
        print_failure("✗ Some verifications failed.")
        print("\nPlease fix the issues above before proceeding.")

    return results


def generate_report(results: Dict[str, Tuple[bool, str]], output_path: str):
    """生成Markdown验证报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    python_version = sys.version.split()[0]

    report = f"""# LLM-for-OR 验证报告

**验证时间**: {timestamp}
**Python版本**: {python_version}

## 验证结果

| 模块 | 状态 | 说明 |
|------|------|------|
"""

    for name, (success, message) in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        report += f"| {name} | {status} | {message} |\n"

    total = len(results)
    passed = sum(1 for success, _ in results.values() if success)

    report += f"""
## 总结

- **总计**: {total} 项检查
- **通过**: {passed} 项
- **失败**: {total - passed} 项
- **成功率**: {passed/total*100:.1f}%

## 建议

"""

    if passed == total:
        report += """✓ 所有验证通过！您可以：
- 运行实验 (`python demo/quickstart.py`)
- 下载 MIPLIB 数据集
- 训练 LLM 代理
"""
    else:
        report += """✗ 部分验证失败。请检查上述问题并重新运行验证。

常见问题：
1. 确保安装了 Gurobi 并配置了有效许可证
2. 检查所有必需的 Python 包已安装
3. 运行 `pytest tests/unit/` 查看详细测试结果
"""

    # 写入文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify LLM-for-OR installation"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output path for verification report (Markdown)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed error messages",
    )

    args = parser.parse_args()

    # 运行验证
    results = run_verification(verbose=args.verbose)

    # 生成报告
    if args.report:
        generate_report(results, args.report)

    # 返回退出码
    all_passed = all(success for success, _ in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
