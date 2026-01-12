"""
Robust方法成功率验证脚本

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-12_robust_injection.md

验证四种Error Type的Basic vs Robust方法成功率对比：
- Type A: 预期 basic 20-40% vs robust 70-85%
- Type B: 预期 basic 40-60% vs robust 80-90%
- Type C: 预期 basic 30-50% vs robust 50-70%
- Type D: 预期 basic ~100% vs robust ~100%

Usage:
    python scripts/validate_robust_methods.py --n_trials 30
    python scripts/validate_robust_methods.py --n_trials 50 --verbose

Author: Ruicheng Ao
Created: 2026-01-12
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import gurobipy as gp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import SaboteurAgent, ErrorType
from src.solvers import GurobiSolver


def create_test_lp(seed: int, problem_id: str = "test_lp") -> gp.Model:
    """
    创建用于测试的随机可行LP问题（紧约束版本）

    Args:
        seed: 随机种子
        problem_id: 问题ID

    Returns:
        可行的Gurobi Model对象
    """
    np.random.seed(seed)
    m = gp.Model(problem_id)
    m.Params.OutputFlag = 0

    n_vars = np.random.randint(6, 12)
    n_constraints = np.random.randint(5, 12)

    # 变量: x_i in [0, 10]
    x = [m.addVar(lb=0, ub=10, name=f"x{i}") for i in range(n_vars)]

    # 添加一些紧约束（使得翻转更可能导致infeasibility）
    # 策略1: 使用较小的RHS使约束更紧
    for i in range(n_constraints):
        coeffs = np.random.uniform(0.5, 2.0, n_vars)
        # RHS设置: 使用更紧的边界
        if i < n_constraints // 2:
            # 紧约束: RHS接近最小可行值
            rhs = np.sum(coeffs * 5) * np.random.uniform(0.8, 1.2)
            expr = gp.quicksum(coeffs[j] * x[j] for j in range(n_vars))
            m.addConstr(expr <= rhs, name=f"c{i}")
        else:
            # 混合约束: 有些是>=
            rhs = np.sum(coeffs * 2) * np.random.uniform(0.5, 0.9)
            expr = gp.quicksum(coeffs[j] * x[j] for j in range(n_vars))
            m.addConstr(expr >= rhs, name=f"c{i}")

    # 添加额外的关键约束使模型更受限
    key_coeffs = np.random.uniform(1.0, 2.0, n_vars)
    key_rhs = np.sum(key_coeffs * 5)  # 恰好在中间
    expr = gp.quicksum(key_coeffs[j] * x[j] for j in range(n_vars))
    m.addConstr(expr <= key_rhs, name="c_key_upper")
    m.addConstr(expr >= key_rhs * 0.5, name="c_key_lower")

    # 目标函数: 最大化，使变量趋向于上限
    obj_coeffs = np.random.uniform(0.5, 1.5, n_vars)
    m.setObjective(
        gp.quicksum(obj_coeffs[j] * x[j] for j in range(n_vars)),
        gp.GRB.MAXIMIZE
    )

    m.update()
    return m


def create_test_mip(seed: int, problem_id: str = "test_mip") -> gp.Model:
    """
    创建用于测试的随机可行MIP问题（紧约束版本）

    Args:
        seed: 随机种子
        problem_id: 问题ID

    Returns:
        可行的Gurobi Model对象
    """
    np.random.seed(seed)
    m = gp.Model(problem_id)
    m.Params.OutputFlag = 0

    n_vars = np.random.randint(8, 15)
    n_constraints = np.random.randint(6, 14)
    n_integer = max(3, int(n_vars * np.random.uniform(0.4, 0.6)))

    # 变量: 前n_integer个为整数，其余为连续
    x = []
    for i in range(n_vars):
        if i < n_integer:
            # 整数变量，上限较大以便Type B测试
            x.append(m.addVar(lb=0, ub=15, vtype=gp.GRB.INTEGER, name=f"x{i}"))
        else:
            x.append(m.addVar(lb=0, ub=10, vtype=gp.GRB.CONTINUOUS, name=f"x{i}"))

    # 添加紧约束
    for i in range(n_constraints):
        coeffs = np.random.uniform(0.5, 2.0, n_vars)
        if i < n_constraints // 2:
            # 紧的上限约束
            rhs = np.sum(coeffs * 5) * np.random.uniform(0.9, 1.3)
            expr = gp.quicksum(coeffs[j] * x[j] for j in range(n_vars))
            m.addConstr(expr <= rhs, name=f"c{i}")
        else:
            # 下限约束
            rhs = np.sum(coeffs * 2) * np.random.uniform(0.4, 0.8)
            expr = gp.quicksum(coeffs[j] * x[j] for j in range(n_vars))
            m.addConstr(expr >= rhs, name=f"c{i}")

    # 添加关键约束
    key_coeffs = np.random.uniform(1.0, 2.0, n_vars)
    key_rhs = np.sum(key_coeffs * 5)
    expr = gp.quicksum(key_coeffs[j] * x[j] for j in range(n_vars))
    m.addConstr(expr <= key_rhs, name="c_key_upper")
    m.addConstr(expr >= key_rhs * 0.4, name="c_key_lower")

    # 目标函数
    obj_coeffs = np.random.uniform(0.5, 1.5, n_vars)
    m.setObjective(
        gp.quicksum(obj_coeffs[j] * x[j] for j in range(n_vars)),
        gp.GRB.MAXIMIZE
    )

    m.update()
    return m


def test_single_injection(
    error_type: str,
    use_robust: bool,
    seed: int,
    use_mip: bool = False
) -> Tuple[bool, str]:
    """
    测试单次注入

    Args:
        error_type: 错误类型 (A/B/C/D)
        use_robust: 是否使用robust方法
        seed: 随机种子
        use_mip: 是否使用MIP模型

    Returns:
        (success, message) tuple
    """
    try:
        # 创建模型
        if use_mip:
            model = create_test_mip(seed, f"test_{error_type}_{seed}")
        else:
            model = create_test_lp(seed, f"test_{error_type}_{seed}")

        # 验证原模型可行
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            return False, "Original model not optimal"

        # 创建新模型用于注入（避免修改原模型）
        if use_mip:
            model = create_test_mip(seed, f"test_{error_type}_{seed}")
        else:
            model = create_test_lp(seed, f"test_{error_type}_{seed}")

        solver = GurobiSolver(model)
        saboteur = SaboteurAgent(solver, seed=seed)

        # 执行注入
        if use_robust:
            result = saboteur.inject_error_robust(error_type)
        else:
            result = saboteur.inject_error(error_type)

        if result.success:
            return True, f"IIS size: {result.iis_size}"
        else:
            return False, "Injection failed"

    except Exception as e:
        return False, f"Error: {str(e)}"


def validate_type_success_rates(
    n_trials: int = 20,
    verbose: bool = False
) -> Dict:
    """
    验证各类型的成功率

    Args:
        n_trials: 每种类型的测试次数
        verbose: 是否打印详细信息

    Returns:
        包含成功率对比的字典
    """
    results = {}
    error_types = ['A', 'B', 'C', 'D']

    # 预期成功率阈值
    expected_rates = {
        'A': {'basic': (0.20, 0.40), 'robust': (0.70, 0.85)},
        'B': {'basic': (0.40, 0.60), 'robust': (0.80, 0.90)},
        'C': {'basic': (0.30, 0.50), 'robust': (0.50, 0.70)},
        'D': {'basic': (0.95, 1.00), 'robust': (0.95, 1.00)},
    }

    for error_type in error_types:
        print(f"\n{'='*60}")
        print(f"Testing Type {error_type}...")
        print(f"{'='*60}")

        basic_success = 0
        robust_success = 0
        basic_details = []
        robust_details = []

        # 对LP和MIP都测试
        for trial in range(n_trials):
            # 使用不同种子
            seed = 42 + trial * 100 + ord(error_type)

            # LP测试 (60%的trial用LP)
            use_mip = trial >= int(n_trials * 0.6)

            # Basic方法
            success, msg = test_single_injection(error_type, use_robust=False, seed=seed, use_mip=use_mip)
            if success:
                basic_success += 1
            basic_details.append((trial, success, msg))

            # Robust方法（使用不同种子避免重复）
            seed_robust = seed + 1000
            success, msg = test_single_injection(error_type, use_robust=True, seed=seed_robust, use_mip=use_mip)
            if success:
                robust_success += 1
            robust_details.append((trial, success, msg))

            if verbose:
                model_type = "MIP" if use_mip else "LP"
                print(f"  Trial {trial+1:2d} ({model_type}): Basic={'OK' if basic_details[-1][1] else 'FAIL'}, "
                      f"Robust={'OK' if robust_details[-1][1] else 'FAIL'}")

        basic_rate = basic_success / n_trials
        robust_rate = robust_success / n_trials

        # 检查是否符合预期
        expected_basic = expected_rates[error_type]['basic']
        expected_robust = expected_rates[error_type]['robust']

        basic_ok = expected_basic[0] <= basic_rate <= expected_basic[1]
        robust_ok = robust_rate >= expected_robust[0]

        results[error_type] = {
            'basic_success': basic_success,
            'basic_rate': basic_rate,
            'basic_expected': expected_basic,
            'basic_ok': basic_ok,
            'robust_success': robust_success,
            'robust_rate': robust_rate,
            'robust_expected': expected_robust,
            'robust_ok': robust_ok,
            'improvement': robust_rate - basic_rate,
        }

        print(f"\nType {error_type} Results:")
        print(f"  Basic:  {basic_rate*100:5.1f}% ({basic_success}/{n_trials}) "
              f"[expected: {expected_basic[0]*100:.0f}-{expected_basic[1]*100:.0f}%] "
              f"{'✓' if basic_ok else '!'}")
        print(f"  Robust: {robust_rate*100:5.1f}% ({robust_success}/{n_trials}) "
              f"[expected: >={expected_robust[0]*100:.0f}%] "
              f"{'✓' if robust_ok else '✗'}")
        print(f"  Improvement: {(robust_rate - basic_rate)*100:+.1f}%")

    return results


def print_summary_report(results: Dict, n_trials: int):
    """打印汇总报告"""
    print("\n" + "="*70)
    print("ROBUST METHOD VALIDATION REPORT")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trials per type: {n_trials}")
    print("-"*70)

    all_passed = True

    for error_type in ['A', 'B', 'C', 'D']:
        r = results[error_type]
        status = "✓ PASS" if r['robust_ok'] else "✗ FAIL"
        all_passed = all_passed and r['robust_ok']

        print(f"\nType {error_type}: {status}")
        print(f"  Basic:       {r['basic_rate']*100:5.1f}% ({r['basic_success']}/{n_trials})")
        print(f"  Robust:      {r['robust_rate']*100:5.1f}% ({r['robust_success']}/{n_trials})")
        print(f"  Improvement: {r['improvement']*100:+5.1f}%")
        print(f"  Threshold:   >= {r['robust_expected'][0]*100:.0f}%")

    print("\n" + "="*70)
    if all_passed:
        print("OVERALL: ✓ ALL TYPES PASSED - Robust methods working as expected")
    else:
        print("OVERALL: ✗ SOME TYPES FAILED - Review and debug required")
        failed = [t for t in ['A', 'B', 'C', 'D'] if not results[t]['robust_ok']]
        print(f"         Failed types: {', '.join(failed)}")
    print("="*70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="验证Robust注入方法成功率")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="每种类型的测试次数 (default: 20)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细输出")
    parser.add_argument("--output", type=str, default=None,
                        help="保存结果到JSON文件")

    args = parser.parse_args()

    print("="*70)
    print("ROBUST METHOD VALIDATION")
    print("="*70)
    print(f"Running {args.n_trials} trials per error type...")
    print(f"Testing both LP and MIP models (60% LP, 40% MIP)")

    # 运行验证
    results = validate_type_success_rates(
        n_trials=args.n_trials,
        verbose=args.verbose
    )

    # 打印汇总
    all_passed = print_summary_report(results, args.n_trials)

    # 保存结果
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'date': datetime.now().isoformat(),
            'n_trials': args.n_trials,
            'all_passed': all_passed,
            'results': {
                t: {
                    'basic_rate': r['basic_rate'],
                    'robust_rate': r['robust_rate'],
                    'improvement': r['improvement'],
                    'passed': r['robust_ok']
                }
                for t, r in results.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # 返回状态码
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
