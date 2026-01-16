"""
批量数据集生成脚本 - 使用SaboteurAgent生成合成测试数据

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A3_Data_Generation.md

This script generates synthetic OR debugging datasets by:
1. Creating simple feasible LP/MIP optimization models
2. Using SaboteurAgent to inject controlled errors (Type A-I)
3. Saving models as MPS files and metadata as JSON
4. Generating dataset statistics and validation reports

Error Types:
    - A-D: Standard errors (single-fix, IIS visible)
    - E: Multi-Constraint Conflict (requires 2+ fixes)
    - F: Hidden Dependency (root cause not in IIS)
    - G: Cascading Conflict (fixing one causes another) [MDP-advantage]
    - H: IIS-Incomplete (IIS shows symptom, not root cause) [MDP-advantage]
    - I: Optimal Selection (multiple fixes, one preserves objective) [MDP-advantage]

Usage:
    # Generate 20 problems with standard error types
    python scripts/data_generation/generate_dataset.py --n_problems 20 --output data/synthetic/debug_bench_v1

    # Generate with hard problems (Type E/F)
    python scripts/data_generation/generate_dataset.py --error_types A,B,C,D,E,F --n_problems 60

    # Generate only MDP-advantage problems (Type G/H/I)
    python scripts/data_generation/generate_dataset.py --error_types G,H,I --n_problems 30

    # Generate with --include_mdp flag for all types including G/H/I
    python scripts/data_generation/generate_dataset.py --include_mdp --n_problems 100

    # Generate difficulty-stratified benchmarks (Phase 3+)
    python scripts/data_generation/generate_dataset.py --difficulty easy --n_problems 500 --output data/benchmarks/or_debug_bench_easy
    python scripts/data_generation/generate_dataset.py --difficulty medium --n_problems 500 --output data/benchmarks/or_debug_bench_medium
    python scripts/data_generation/generate_dataset.py --difficulty difficult --n_problems 500 --output data/benchmarks/or_debug_bench_difficult

    # Validate existing dataset
    python scripts/data_generation/generate_dataset.py --validate data/synthetic/debug_bench_v1/dataset.json

Author: Ruicheng Ao
Created: 2026-01-11
Updated: 2026-01-16 (Added --difficulty flag for stratified benchmark generation)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_generation import SaboteurAgent, ErrorType, Difficulty, ProblemValidator
from src.data_generation.difficulty_generator import DifficultyConfig, classify_problem_difficulty
from src.solvers import GurobiSolver


def create_simple_lp(
    n_vars: int,
    n_constraints: int,
    seed: int,
    problem_id: str
) -> gp.Model:
    """
    创建紧约束版本的随机可行LP问题

    使用更紧的约束边界，使得Type A翻转更可能导致infeasibility。

    Args:
        n_vars: 变量数量
        n_constraints: 约束数量
        seed: 随机种子
        problem_id: 问题ID

    Returns:
        可行的Gurobi Model对象
    """
    np.random.seed(seed)
    m = gp.Model(problem_id)
    m.Params.OutputFlag = 0

    # 变量: x_i in [0, 10]
    x = [m.addVar(lb=0, ub=10, name=f"x{i}") for i in range(n_vars)]

    # 混合约束: 一半<=约束，一半>=约束（使翻转更有效）
    for i in range(n_constraints):
        coeffs = np.random.uniform(0.5, 2.0, n_vars)
        if i < n_constraints // 2:
            # 紧的上限约束
            rhs = np.sum(coeffs * 5) * np.random.uniform(0.8, 1.2)
            expr = gp.quicksum(coeffs[j] * x[j] for j in range(n_vars))
            m.addConstr(expr <= rhs, name=f"c{i}")
        else:
            # 下限约束
            rhs = np.sum(coeffs * 2) * np.random.uniform(0.5, 0.9)
            expr = gp.quicksum(coeffs[j] * x[j] for j in range(n_vars))
            m.addConstr(expr >= rhs, name=f"c{i}")

    # 添加关键约束对（使模型更受限）
    key_coeffs = np.random.uniform(1.0, 2.0, n_vars)
    key_rhs = np.sum(key_coeffs * 5)
    expr = gp.quicksum(key_coeffs[j] * x[j] for j in range(n_vars))
    m.addConstr(expr <= key_rhs, name="c_key_upper")
    m.addConstr(expr >= key_rhs * 0.5, name="c_key_lower")

    # 目标函数: max sum(c_i * x_i)
    obj_coeffs = np.random.uniform(0.5, 1.5, n_vars)
    m.setObjective(
        gp.quicksum(obj_coeffs[j] * x[j] for j in range(n_vars)),
        gp.GRB.MAXIMIZE
    )

    m.update()
    return m


def create_simple_mip(
    n_vars: int,
    n_constraints: int,
    n_integer: int,
    seed: int,
    problem_id: str
) -> gp.Model:
    """
    创建紧约束版本的随机可行MIP问题

    使用更紧的约束边界，使得注入更可能导致infeasibility。

    Args:
        n_vars: 变量数量
        n_constraints: 约束数量
        n_integer: 整数变量数量
        seed: 随机种子
        problem_id: 问题ID

    Returns:
        可行的Gurobi Model对象
    """
    np.random.seed(seed)
    m = gp.Model(problem_id)
    m.Params.OutputFlag = 0

    # 变量: 前n_integer个为整数（上限15便于Type B测试），其余为连续
    x = []
    for i in range(n_vars):
        if i < n_integer:
            x.append(m.addVar(lb=0, ub=15, vtype=gp.GRB.INTEGER, name=f"x{i}"))
        else:
            x.append(m.addVar(lb=0, ub=10, vtype=gp.GRB.CONTINUOUS, name=f"x{i}"))

    # 混合约束: 一半<=约束，一半>=约束
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

    # 添加关键约束对
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


def generate_problem(
    problem_idx: int,
    error_type: str,
    output_dir: Path,
    seed: int,
    problem_type: str = "lp",
    use_robust: bool = True,
    target_iis_size: int = None
) -> Dict:
    """
    生成单个问题：创建模型 -> 注入错误 -> 保存

    Args:
        problem_idx: 问题索引
        error_type: 错误类型 (A/B/C/D)
        output_dir: 输出目录
        seed: 随机种子
        problem_type: 问题类型 (lp/mip)
        use_robust: 是否使用robust注入方法 (default: True)
        target_iis_size: Type D的目标IIS大小 (用于难度控制)

    Returns:
        问题元数据字典
    """
    # 生成problem_id
    problem_id = f"{problem_type}_type{error_type}_{problem_idx:03d}"

    # 创建基础模型
    if problem_type == "lp":
        # LP: 5-10变量, 3-8约束
        n_vars = np.random.randint(5, 11)
        n_constraints = np.random.randint(3, 9)
        model = create_simple_lp(n_vars, n_constraints, seed, problem_id)
    else:  # mip
        # MIP: 8-15变量, 5-12约束, 30-50%整数变量
        n_vars = np.random.randint(8, 16)
        n_constraints = np.random.randint(5, 13)
        n_integer = max(1, int(n_vars * np.random.uniform(0.3, 0.5)))
        model = create_simple_mip(n_vars, n_constraints, n_integer, seed, problem_id)

    # 保存原始模型副本（用于验证）
    original_model = model.copy()

    # 验证原始模型可行
    solver = GurobiSolver.from_model(model)
    original_state = solver.solve()

    if original_state.status != "OPTIMAL":
        print(f"  Warning: {problem_id} 原始模型非最优: {original_state.status}")
        return None

    # 使用SaboteurAgent注入错误
    saboteur = SaboteurAgent(solver, seed=seed + 1000)

    if use_robust:
        # 使用robust方法
        if error_type == "D" and target_iis_size is not None:
            result = saboteur.inject_type_d_robust(target_iis_size=target_iis_size)
        else:
            result = saboteur.inject_error_robust(error_type)
    else:
        result = saboteur.inject_error(error_type)

    if not result.success:
        return None

    # 保存MPS文件
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"{problem_id}.mps"
    solver._model.write(str(model_file))

    # 生成问题自然语言描述
    problem_nl = generate_problem_description(
        problem_id, error_type, n_vars,
        len(solver.get_all_constraints())
    )

    # 使用InjectionResult中的IIS信息
    iis_constraints = result.iis_constraints if result.iis_constraints else []
    iis_bounds = result.iis_bounds if result.iis_bounds else []
    iis_size = result.iis_size

    # 如果InjectionResult没有IIS信息，重新计算
    if not iis_constraints and result.solver_status in ["INFEASIBLE", "INF_OR_UNBD"]:
        try:
            iis = solver.compute_iis()
            iis_constraints = iis.constraints
            iis_bounds = iis.bounds
            iis_size = iis.size
        except Exception as e:
            print(f"  Warning: {problem_id} IIS计算失败: {e}")

    # 获取难度分类
    difficulty = result.difficulty.value if hasattr(result, 'difficulty') and result.difficulty else Difficulty.from_iis_size(iis_size).value

    # 构建元数据
    metadata = {
        "problem_id": problem_id,
        "model_file": str(model_file.relative_to(output_dir)),
        "problem_nl": problem_nl,
        "error_type": error_type,
        "error_description": result.metadata.get("description", "") if result.metadata else "",
        "ground_truth_fix": result.ground_truth_fix,
        "target_name": result.target_name,
        "target_type": result.metadata.get("target_type", "constraint") if result.metadata else "constraint",
        "initial_status": result.solver_status,
        "original_status": original_state.status,
        "original_objective": result.original_objective,
        "n_variables": n_vars,
        "n_constraints": len(solver.get_all_constraints()),
        "iis_constraints": iis_constraints,
        "iis_bounds": iis_bounds,
        "iis_size": iis_size,
        "difficulty": difficulty,
        "problem_type": problem_type,
        "use_robust": use_robust,
        "created_at": datetime.now().isoformat()
    }

    return metadata


def generate_problem_description(
    problem_id: str,
    error_type: str,
    n_vars: int,
    n_constraints: int
) -> str:
    """生成问题的自然语言描述"""
    error_names = {
        "A": "Constraint Direction Error",
        "B": "Variable Type Error",
        "C": "Logic Error (Missing Term)",
        "D": "Conflicting Constraint",
        "E": "Multi-Constraint Conflict (requires 2+ fixes)",
        "F": "Hidden Dependency (root cause not directly visible)",
        "G": "Cascading Conflict (fixing one constraint causes another to conflict)",
        "H": "IIS-Incomplete (IIS shows symptom, not root cause)",
        "I": "Optimal Selection (multiple fixes possible, one preserves objective)"
    }

    error_name = error_names.get(error_type, 'Unknown')

    # Type E/F are harder problems, Type G/H/I are MDP-advantage problems
    if error_type in ["G", "H", "I"]:
        return (
            f"Linear programming problem {problem_id} with {n_vars} variables "
            f"and {n_constraints} constraints. Contains a {error_name} "
            f"that makes the model infeasible. This is an MDP-advantage problem "
            f"requiring strategic multi-step reasoning - single-step repairs will fail."
        )
    elif error_type in ["E", "F"]:
        return (
            f"Linear programming problem {problem_id} with {n_vars} variables "
            f"and {n_constraints} constraints. Contains a {error_name} "
            f"that makes the model infeasible. This is a challenging problem "
            f"requiring multi-step reasoning to diagnose and fix."
        )
    else:
        return (
            f"Linear programming problem {problem_id} with {n_vars} variables "
            f"and {n_constraints} constraints. Contains a {error_name} "
            f"that makes the model infeasible."
        )


def generate_dataset(
    n_problems: int,
    error_types: List[str],
    output_dir: str,
    problem_types: List[str],
    seed: int,
    use_robust: bool = True,
    difficulty_config: "DifficultyConfig" = None
) -> Dict:
    """
    批量生成数据集

    Args:
        n_problems: 问题总数
        error_types: 错误类型列表
        output_dir: 输出目录
        problem_types: 问题类型列表 (lp/mip)
        seed: 随机种子
        use_robust: 是否使用robust注入方法 (default: True)
        difficulty_config: Optional difficulty configuration for IIS size control

    Returns:
        数据集字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_problems} problems...")
    print(f"Error types: {error_types}")
    print(f"Problem types: {problem_types}")
    print(f"Use robust methods: {use_robust}")
    print(f"Output directory: {output_dir}")
    print()

    # 分配错误类型
    problems_per_type = n_problems // len(error_types)
    assigned_types = []
    for et in error_types:
        assigned_types.extend([et] * problems_per_type)

    # 补充剩余问题
    remaining = n_problems - len(assigned_types)
    for i in range(remaining):
        assigned_types.append(error_types[i % len(error_types)])

    # 生成问题
    problems = []
    failed = 0
    success_by_type = {et: 0 for et in error_types}
    fail_by_type = {et: 0 for et in error_types}

    for i, error_type in enumerate(assigned_types):
        # 选择问题类型（70% LP, 30% MIP）
        problem_type = np.random.choice(problem_types, p=[0.7, 0.3])

        # 确定目标IIS大小（基于难度配置或默认行为）
        target_iis_size = None
        if difficulty_config:
            # Use difficulty-specific IIS range
            iis_min, iis_max = difficulty_config.iis_range
            target_iis_size = np.random.randint(iis_min, iis_max + 1)
        elif error_type == "D":
            # 对于Type D，随机选择目标IIS大小以控制难度
            # 30% easy (2), 45% medium (4), 25% hard (6)
            target_iis_size = np.random.choice([2, 4, 6], p=[0.3, 0.45, 0.25])

        print(f"[{i+1}/{n_problems}] Generating {problem_type.upper()} Type {error_type}...", end=" ")

        try:
            metadata = generate_problem(
                i, error_type, output_path,
                seed=seed + i,
                problem_type=problem_type,
                use_robust=use_robust,
                target_iis_size=target_iis_size
            )

            if metadata is None:
                print("FAILED")
                failed += 1
                fail_by_type[error_type] += 1
            else:
                problems.append(metadata)
                success_by_type[error_type] += 1
                difficulty = metadata.get('difficulty', 'unknown')
                print(f"OK ({metadata['initial_status']}, {difficulty})")

        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
            fail_by_type[error_type] += 1

    # 计算成功率
    success_rates = {}
    for et in error_types:
        total = success_by_type[et] + fail_by_type[et]
        success_rates[et] = success_by_type[et] / total if total > 0 else 0

    # 计算难度分布
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    for p in problems:
        diff = p.get("difficulty", "medium")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    # 构建数据集
    difficulty_level = difficulty_config.level.value if difficulty_config else None
    dataset = {
        "dataset_name": output_path.name,
        "description": "Synthetic OR debugging benchmark generated with SaboteurAgent (robust methods)",
        "created_at": datetime.now().isoformat(),
        "num_problems": len(problems),
        "num_failed": failed,
        "error_types": error_types,
        "problem_types": problem_types,
        "seed": seed,
        "use_robust": use_robust,
        "difficulty_level": difficulty_level,
        "difficulty_iis_range": list(difficulty_config.iis_range) if difficulty_config else None,
        "success_by_type": success_by_type,
        "fail_by_type": fail_by_type,
        "success_rates": success_rates,
        "difficulty_distribution": difficulty_counts,
        "problems": problems
    }

    # 保存dataset.json
    dataset_file = output_path / "dataset.json"
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    print()
    print(f"✓ Dataset saved to {dataset_file}")
    print(f"✓ Successfully generated: {len(problems)}")
    print(f"✗ Failed: {failed}")

    # 生成统计报告
    generate_report(dataset, output_path)

    return dataset


def generate_report(dataset: Dict, output_dir: Path):
    """生成数据集统计报告"""
    report_file = output_dir / "report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"Dataset Report: {dataset['dataset_name']}\n")
        f.write("="*60 + "\n\n")

        f.write(f"Created: {dataset['created_at']}\n")
        f.write(f"Total Problems: {dataset['num_problems']}\n")
        f.write(f"Failed: {dataset['num_failed']}\n")
        f.write(f"Use Robust Methods: {dataset.get('use_robust', False)}\n\n")

        # 成功率统计
        f.write("Success Rates by Type:\n")
        success_rates = dataset.get('success_rates', {})
        for et, rate in sorted(success_rates.items()):
            success = dataset.get('success_by_type', {}).get(et, 0)
            fail = dataset.get('fail_by_type', {}).get(et, 0)
            f.write(f"  Type {et}: {rate*100:.1f}% ({success}/{success + fail})\n")
        f.write("\n")

        # 错误类型统计
        error_counts = {}
        for p in dataset['problems']:
            et = p['error_type']
            error_counts[et] = error_counts.get(et, 0) + 1

        f.write("Error Type Distribution:\n")
        for et, count in sorted(error_counts.items()):
            f.write(f"  Type {et}: {count}\n")
        f.write("\n")

        # 难度分布
        f.write("Difficulty Distribution:\n")
        difficulty_dist = dataset.get('difficulty_distribution', {})
        total = sum(difficulty_dist.values())
        for diff, count in sorted(difficulty_dist.items()):
            pct = count / total * 100 if total > 0 else 0
            f.write(f"  {diff}: {count} ({pct:.1f}%)\n")
        f.write("\n")

        # 状态统计
        status_counts = {}
        for p in dataset['problems']:
            st = p['initial_status']
            status_counts[st] = status_counts.get(st, 0) + 1

        f.write("Initial Status Distribution:\n")
        for st, count in sorted(status_counts.items()):
            f.write(f"  {st}: {count}\n")
        f.write("\n")

        # IIS大小统计
        iis_sizes = [p.get('iis_size', 0) for p in dataset['problems']]
        if iis_sizes:
            f.write("IIS Size:\n")
            f.write(f"  min={min(iis_sizes)}, max={max(iis_sizes)}, avg={np.mean(iis_sizes):.1f}\n")
            f.write("\n")

        # 规模统计
        vars_list = [p['n_variables'] for p in dataset['problems']]
        constrs_list = [p['n_constraints'] for p in dataset['problems']]

        f.write("Problem Size:\n")
        f.write(f"  Variables: min={min(vars_list)}, max={max(vars_list)}, avg={np.mean(vars_list):.1f}\n")
        f.write(f"  Constraints: min={min(constrs_list)}, max={max(constrs_list)}, avg={np.mean(constrs_list):.1f}\n")

    print(f"✓ Report saved to {report_file}")


def validate_dataset(dataset_path: str):
    """验证数据集完整性"""
    print(f"Validating dataset: {dataset_path}")

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"  Dataset: {dataset['dataset_name']}")
    print(f"  Problems: {dataset['num_problems']}")

    # 检查文件存在性
    base_dir = Path(dataset_path).parent
    missing = 0

    for p in dataset['problems']:
        model_file = base_dir / p['model_file']
        if not model_file.exists():
            print(f"  ✗ Missing: {p['model_file']}")
            missing += 1

    if missing == 0:
        print(f"  ✓ All {len(dataset['problems'])} MPS files found")
    else:
        print(f"  ✗ Missing {missing} files")

    # 检查必需字段
    required_fields = [
        'problem_id', 'model_file', 'error_type',
        'ground_truth_fix', 'initial_status'
    ]

    for p in dataset['problems']:
        for field in required_fields:
            if field not in p:
                print(f"  ✗ Missing field '{field}' in {p.get('problem_id', 'unknown')}")

    print("  ✓ Validation complete")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OR debugging dataset"
    )

    parser.add_argument(
        "--n_problems",
        type=int,
        default=20,
        help="Number of problems to generate (default: 20)"
    )

    parser.add_argument(
        "--error_types",
        type=str,
        default="A,B,C,D",
        help="Comma-separated error types (default: A,B,C,D). Use A-I for all types."
    )

    parser.add_argument(
        "--include_hard",
        action="store_true",
        help="Include hard problem types (E: Multi-Constraint, F: Hidden Dependency)"
    )

    parser.add_argument(
        "--include_mdp",
        action="store_true",
        help="Include MDP-advantage problem types (G: Cascading, H: IIS-Incomplete, I: Optimal Selection)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/debug_bench_v1",
        help="Output directory (default: data/synthetic/debug_bench_v1)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--problem_types",
        type=str,
        default="lp,mip",
        help="Comma-separated problem types (default: lp,mip)"
    )

    parser.add_argument(
        "--validate",
        type=str,
        help="Validate existing dataset.json file"
    )

    parser.add_argument(
        "--use_robust",
        action="store_true",
        default=True,
        help="Use robust injection methods (default: True)"
    )

    parser.add_argument(
        "--no_robust",
        action="store_true",
        help="Disable robust injection methods"
    )

    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "difficult"],
        default=None,
        help="Generate problems of specified difficulty level. "
             "Overrides --error_types with difficulty-appropriate types. "
             "(easy: A-C, IIS 1-2; medium: D-F, IIS 3-5; difficult: E,G-I, IIS 5-10)"
    )

    args = parser.parse_args()

    # 验证模式
    if args.validate:
        validate_dataset(args.validate)
        return

    # 确定是否使用robust方法
    use_robust = not args.no_robust

    # 生成模式
    difficulty_config = None
    if args.difficulty:
        # Use difficulty-stratified generation
        difficulty_config = DifficultyConfig.get_config(args.difficulty)
        error_types = difficulty_config.error_types
        print(f"Using difficulty level: {args.difficulty}")
        print(f"  Error types: {error_types}")
        print(f"  IIS range: {difficulty_config.iis_range}")
        print(f"  Expected SFT RR@5: {difficulty_config.expected_rr5[0]*100:.0f}%-{difficulty_config.expected_rr5[1]*100:.0f}%")
        print()
    else:
        error_types = args.error_types.split(',')

        # 如果指定include_hard，添加E和F类型
        if args.include_hard and 'E' not in error_types:
            error_types.extend(['E', 'F'])

        # 如果指定include_mdp，添加G, H, I类型
        if args.include_mdp:
            for t in ['G', 'H', 'I']:
                if t not in error_types:
                    error_types.append(t)

    problem_types = args.problem_types.split(',')

    dataset = generate_dataset(
        n_problems=args.n_problems,
        error_types=error_types,
        output_dir=args.output,
        problem_types=problem_types,
        seed=args.seed,
        use_robust=use_robust,
        difficulty_config=difficulty_config
    )

    print()
    print("✓ Dataset generation complete!")
    print(f"  Use: python scripts/run_llm_experiment.py --dataset {args.output}/dataset.json")


if __name__ == "__main__":
    main()
