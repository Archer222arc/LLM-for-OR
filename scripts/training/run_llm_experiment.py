#!/usr/bin/env python3
"""
LLM实验运行脚本

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/

功能：
    - 加载实验配置（YAML）
    - 加载数据集（JSON）
    - 初始化各类Agent（LLM + Baseline）
    - 运行BenchmarkRunner评估（支持并行模式）
    - 生成对比报告（Markdown + JSON）
    - 保存实验轨迹和日志

并行模式:
    - 使用ProcessPoolExecutor实现Agent级别并行
    - 每个Agent在独立进程中运行，避免Gurobi线程安全问题
    - 预期加速: ~10x (12 Agent × 20 Problem 从30分钟到3分钟)

Usage:
    # 运行完整实验（顺序模式）
    python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml

    # 并行模式（推荐）
    python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml --parallel --max-workers 4

    # 只运行指定agent
    python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml --agents gpt4,heuristic

    # 限制问题数量（用于快速测试）
    python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml --limit 2 --parallel

    # 验证配置文件
    python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml --validate-only

Dependencies:
    - src.agents: LLMAgent, HeuristicAgent, RandomAgent, GreedyDropAgent, DoNothingAgent
    - src.evaluation: BenchmarkRunner, BenchmarkConfig, BenchmarkProblem
    - src.solvers: GurobiSolver
    - src.environments: SolverDebugEnv

Created: 2026-01-11
Updated: 2026-01-11 (Added parallel mode)
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents import (
    LLMAgent, HeuristicAgent, RandomAgent,
    GreedyDropAgent, DoNothingAgent, BaseAgent
)
from src.evaluation import BenchmarkRunner, BenchmarkConfig, BenchmarkProblem
from src.solvers import GurobiSolver
from src.environments import SolverDebugEnv


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML格式错误
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"✓ 成功加载配置: {config_path}")
    print(f"  实验名称: {config['experiment']['name']}")
    print(f"  描述: {config['experiment']['description']}")

    return config


def load_dataset_from_json(dataset_path: str, limit: Optional[int] = None) -> List[BenchmarkProblem]:
    """
    从JSON加载数据集，创建BenchmarkProblem列表

    Args:
        dataset_path: 数据集JSON路径
        limit: 限制加载的问题数量（用于快速测试）

    Returns:
        BenchmarkProblem列表

    Example:
        >>> problems = load_dataset_from_json("data/synthetic/debug_bench_v1/dataset.json")
        >>> print(f"加载 {len(problems)} 个问题")
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"\n✓ 成功加载数据集: {dataset['dataset_name']}")
    print(f"  创建时间: {dataset['created_at']}")
    print(f"  问题总数: {dataset['num_problems']}")

    problems = []
    problem_list = dataset['problems'][:limit] if limit else dataset['problems']

    for i, p in enumerate(problem_list):
        try:
            # 加载MPS模型
            model_path = Path(p['model_file'])
            if not model_path.is_absolute():
                # 相对于数据集文件的路径
                model_path = dataset_path.parent / model_path

            if not model_path.exists():
                print(f"  ⚠ 跳过问题 {p['problem_id']}: 模型文件不存在 {model_path}")
                continue

            solver = GurobiSolver.from_file(str(model_path))

            # 创建环境
            env = SolverDebugEnv(
                solver,
                problem_nl=p.get('problem_nl', ''),
                max_steps=50
            )

            # 创建BenchmarkProblem
            problem = BenchmarkProblem(
                problem_id=p['problem_id'],
                env=env,
                ground_truth_fix=p.get('ground_truth_fix'),
                ground_truth_iis=p.get('iis_constraints', []),
                metadata={
                    'error_type': p.get('error_type'),
                    'error_description': p.get('error_description'),
                    'n_variables': p.get('n_variables'),
                    'n_constraints': p.get('n_constraints'),
                    'iis_size': p.get('iis_size')
                }
            )

            problems.append(problem)

            if (i + 1) % 10 == 0:
                print(f"  已加载 {i + 1}/{len(problem_list)} 个问题...")

        except Exception as e:
            print(f"  ⚠ 加载问题 {p['problem_id']} 失败: {e}")
            continue

    print(f"✓ 成功加载 {len(problems)} 个问题\n")

    return problems


def create_agent_from_config(agent_config: Dict[str, Any]) -> BaseAgent:
    """
    从配置创建Agent实例

    Args:
        agent_config: Agent配置字典

    Returns:
        Agent实例

    Supported agent types:
        - llm: LLMAgent (OpenAI, Anthropic, etc.)
        - baseline: HeuristicAgent, RandomAgent, GreedyDropAgent, DoNothingAgent

    Example:
        >>> config = {
        ...     "name": "gpt4",
        ...     "type": "llm",
        ...     "model": "gpt-4",
        ...     "provider": "openai",
        ...     "temperature": 0.0
        ... }
        >>> agent = create_agent_from_config(config)
    """
    agent_type = agent_config['type']
    agent_name = agent_config['name']

    if agent_type == 'llm':
        # 创建LLM Agent
        provider = agent_config['provider']

        # Base parameters for all LLM providers
        llm_params = {
            'model': agent_config['model'],
            'provider': provider,
            'temperature': agent_config.get('temperature', 0.0),
            'max_retries': agent_config.get('max_retries', 3),
            'name': agent_name
        }

        # Add Azure OpenAI specific parameters
        if provider == "azure_openai":
            llm_params['azure_endpoint'] = agent_config.get('azure_endpoint')
            llm_params['api_version'] = agent_config.get('api_version', '2024-10-21')
            llm_params['azure_deployment'] = agent_config.get('azure_deployment')

        agent = LLMAgent(**llm_params)
        print(f"  ✓ 创建LLM Agent: {agent_name} ({agent_config['model']})")

    elif agent_type == 'baseline':
        # 创建Baseline Agent
        cls_name = agent_config['class']

        if cls_name == 'HeuristicAgent':
            agent = HeuristicAgent(name=agent_name)
        elif cls_name == 'RandomAgent':
            seed = agent_config.get('seed', 42)
            agent = RandomAgent(seed=seed, name=agent_name)
        elif cls_name == 'GreedyDropAgent':
            agent = GreedyDropAgent(name=agent_name)
        elif cls_name == 'DoNothingAgent':
            agent = DoNothingAgent(name=agent_name)
        else:
            raise ValueError(f"未知的Baseline Agent类型: {cls_name}")

        print(f"  ✓ 创建Baseline Agent: {agent_name} ({cls_name})")

    else:
        raise ValueError(f"未知的Agent类型: {agent_type}")

    return agent


def generate_markdown_report(
    comparison: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Path
):
    """
    生成Markdown格式的对比报告

    Args:
        comparison: BenchmarkRunner.compare_agents()返回的结果
        config: 实验配置
        output_path: 输出文件路径
    """
    lines = []

    # 判断comparison结构 - 可能是 {agent: metrics} 或 {results: {...}, summary: {...}}
    if 'results' in comparison:
        results = comparison['results']
        n_problems = comparison.get('summary', {}).get('n_problems', 'N/A')
    else:
        # 直接是 {agent_name: metrics} 格式
        results = comparison
        # 从第一个agent的n_episodes推断问题数
        first_metrics = next(iter(results.values()), {})
        n_problems = first_metrics.get('n_episodes', 'N/A')

    # 标题和元数据
    lines.append(f"# {config['experiment']['name']}")
    lines.append(f"\n**描述**: {config['experiment']['description']}")
    lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**数据集**: {config['benchmark']['dataset']}")
    lines.append(f"\n**问题数量**: {n_problems}")
    lines.append("\n---\n")

    # 总体对比表格
    lines.append("## Agent性能对比\n")

    # 检查是否有DA指标
    has_da = any('diagnosis_accuracy' in m for m in results.values())

    if has_da:
        lines.append("| Agent | Recovery Rate | Avg Steps | Diag Accuracy | Diag Precision | Step Efficiency |")
        lines.append("|-------|---------------|-----------|---------------|----------------|-----------------|")
    else:
        lines.append("| Agent | Recovery Rate | Avg Steps | Avg Reward | Success Rate | Step Efficiency |")
        lines.append("|-------|---------------|-----------|------------|--------------|-----------------|")

    for agent_name, metrics in results.items():
        if has_da:
            lines.append(
                f"| {agent_name} | "
                f"{metrics.get('recovery_rate', 0.0):.1%} | "
                f"{metrics.get('avg_steps', 0.0):.2f} | "
                f"{metrics.get('diagnosis_accuracy', 0.0):.1%} | "
                f"{metrics.get('diagnosis_precision', 0.0):.1%} | "
                f"{metrics.get('step_efficiency', 0.0):.2f} |"
            )
        else:
            lines.append(
                f"| {agent_name} | "
                f"{metrics.get('recovery_rate', 0.0):.1%} | "
                f"{metrics.get('avg_steps', 0.0):.2f} | "
                f"{metrics.get('avg_reward', 0.0):.2f} | "
                f"{metrics.get('success_rate', 0.0):.1%} | "
                f"{metrics.get('step_efficiency', 0.0):.2f} |"
            )

    lines.append("\n---\n")

    # 详细指标
    lines.append("## 详细指标\n")
    for agent_name, metrics in results.items():
        lines.append(f"### {agent_name}\n")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- **{metric_name}**: {value:.4f}")
            else:
                lines.append(f"- **{metric_name}**: {value}")
        lines.append("\n")

    # 按错误类型统计（如果有）
    if 'by_error_type' in comparison:
        lines.append("## 按错误类型统计\n")
        for error_type, stats in comparison['by_error_type'].items():
            lines.append(f"### Type {error_type}\n")
            lines.append(f"- 问题数量: {stats['count']}")
            lines.append(f"- 平均恢复率: {stats.get('avg_recovery_rate', 0.0):.1%}\n")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ Markdown报告已保存: {output_path}")


def run_experiment(
    config: Dict[str, Any],
    selected_agents: Optional[List[str]] = None,
    limit: Optional[int] = None,
    parallel: bool = False,
    max_workers: int = 4
):
    """
    运行完整实验

    Args:
        config: 实验配置字典
        selected_agents: 仅运行指定的agent（None表示运行全部）
        limit: 限制问题数量
        parallel: 是否并行运行（每个Agent一个进程）
        max_workers: 并行最大工作进程数
    """
    print("\n" + "="*60)
    print("开始运行LLM实验" + (" [并行模式]" if parallel else " [顺序模式]"))
    print("="*60 + "\n")

    dataset_path = config['benchmark']['dataset']

    if parallel:
        # 并行模式：直接传配置给worker进程
        print("[1/3] 准备Agent配置...")

        agent_configs = []
        for agent_config in config['agents']:
            agent_name = agent_config['name']
            if selected_agents and agent_name not in selected_agents:
                print(f"  ⊘ 跳过 Agent: {agent_name}")
                continue
            agent_configs.append(agent_config)
            print(f"  ✓ 加入Agent: {agent_name}")

        if len(agent_configs) == 0:
            print("❌ 未选择任何Agent，实验终止")
            return

        print(f"\n✓ 共 {len(agent_configs)} 个Agent将并行运行\n")

        # 创建BenchmarkRunner并运行并行评估
        print("[2/3] 并行运行评估...")

        runner = BenchmarkRunner(
            config=BenchmarkConfig(
                max_steps=config['benchmark']['max_steps'],
                n_episodes=config['benchmark'].get('n_episodes', 1),
                verbose=config['benchmark'].get('verbose', True)
            )
        )

        try:
            comparison = runner.compare_agents_parallel(
                dataset_path=dataset_path,
                agent_configs=agent_configs,
                max_workers=max_workers,
                limit=limit,
            )
        except Exception as e:
            print(f"❌ 并行评估过程出错: {e}")
            import traceback
            traceback.print_exc()
            return

        # 生成报告
        print("\n[3/3] 生成报告...")

    else:
        # 顺序模式：原有逻辑
        # 1. 加载数据集
        print("[1/4] 加载数据集...")
        problems = load_dataset_from_json(dataset_path, limit=limit)

        if len(problems) == 0:
            print("❌ 未能加载任何问题，实验终止")
            return

        # 2. 创建Agents
        print("[2/4] 创建Agents...")
        agents = []

        for agent_config in config['agents']:
            agent_name = agent_config['name']

            # 如果指定了selected_agents，只创建选中的
            if selected_agents and agent_name not in selected_agents:
                print(f"  ⊘ 跳过 Agent: {agent_name}")
                continue

            try:
                agent = create_agent_from_config(agent_config)
                agents.append(agent)
            except Exception as e:
                print(f"  ⚠ 创建Agent {agent_name} 失败: {e}")
                continue

        if len(agents) == 0:
            print("❌ 未能创建任何Agent，实验终止")
            return

        print(f"\n✓ 共创建 {len(agents)} 个Agent\n")

        # 3. 创建BenchmarkRunner并运行
        print("[3/4] 运行评估...")

        runner = BenchmarkRunner(
            config=BenchmarkConfig(
                max_steps=config['benchmark']['max_steps'],
                n_episodes=config['benchmark'].get('n_episodes', 1),
                verbose=config['benchmark'].get('verbose', True)
            )
        )

        try:
            comparison = runner.compare_agents(problems, agents)
        except Exception as e:
            print(f"❌ 评估过程出错: {e}")
            import traceback
            traceback.print_exc()
            return

        # 4. 生成报告
        print("\n[4/4] 生成报告...")

    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存JSON结果
    json_path = output_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON结果已保存: {json_path}")

    # 生成Markdown报告
    if 'markdown' in config['evaluation'].get('report_format', ['markdown']):
        md_path = output_dir / 'report.md'
        generate_markdown_report(comparison, config, md_path)

    # 保存轨迹（如果配置）
    if config['evaluation'].get('save_trajectories', False):
        trajectories_dir = output_dir / 'trajectories'
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 轨迹保存目录: {trajectories_dir}")

    print("\n" + "="*60)
    print("实验完成！")
    print("="*60 + "\n")

    # 打印简要结果
    print("📊 实验结果概览:\n")
    # 判断comparison结构
    results = comparison.get('results', comparison) if isinstance(comparison, dict) else comparison
    for agent_name, metrics in results.items():
        print(f"{agent_name}:")
        print(f"  Recovery Rate: {metrics.get('recovery_rate', 0.0):.1%}")
        print(f"  Avg Steps: {metrics.get('avg_steps', 0.0):.2f}")
        print(f"  Success Rate: {metrics.get('success_rate', 0.0):.1%}\n")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件的完整性

    Args:
        config: 配置字典

    Returns:
        是否通过验证
    """
    print("\n验证配置文件...\n")

    required_keys = {
        'experiment': ['name', 'description', 'output_dir'],
        'benchmark': ['dataset', 'max_steps'],
        'agents': [],
        'evaluation': ['metrics']
    }

    valid = True

    # 检查顶层键
    for key in required_keys.keys():
        if key not in config:
            print(f"❌ 缺少必需的配置项: {key}")
            valid = False
        elif key != 'agents':
            for subkey in required_keys[key]:
                if subkey not in config[key]:
                    print(f"❌ 缺少必需的配置项: {key}.{subkey}")
                    valid = False

    # 检查agents配置
    if 'agents' in config:
        if not isinstance(config['agents'], list) or len(config['agents']) == 0:
            print("❌ agents必须是非空列表")
            valid = False
        else:
            for i, agent in enumerate(config['agents']):
                if 'name' not in agent:
                    print(f"❌ agents[{i}]缺少name字段")
                    valid = False
                if 'type' not in agent:
                    print(f"❌ agents[{i}]缺少type字段")
                    valid = False

    # 检查数据集文件
    if 'benchmark' in config and 'dataset' in config['benchmark']:
        dataset_path = Path(config['benchmark']['dataset'])
        if not dataset_path.exists():
            print(f"⚠ 数据集文件不存在: {dataset_path}")
            print("  （运行时会失败，但配置格式正确）")

    if valid:
        print("✓ 配置验证通过\n")
    else:
        print("\n❌ 配置验证失败\n")

    return valid


def main():
    parser = argparse.ArgumentParser(
        description="运行LLM实验评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整实验（顺序模式）
  python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml

  # 并行模式（推荐，~10x加速）
  python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml --parallel --max-workers 4

  # 只运行指定agent
  python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml --agents gpt4,heuristic

  # 限制问题数量（快速测试）
  python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml --limit 2 --parallel

  # 仅验证配置
  python scripts/run_llm_experiment.py --config configs/experiments/llm_eval.yaml --validate-only
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='实验配置文件路径（YAML格式）'
    )

    parser.add_argument(
        '--agents',
        type=str,
        default=None,
        help='仅运行指定的agents（逗号分隔），例如: gpt4,claude3-sonnet'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制问题数量（用于快速测试）'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证配置文件，不运行实验'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='启用并行模式（每个Agent在独立进程中运行）'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='并行模式最大工作进程数（默认4）'
    )

    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config(args.config)

        # 验证配置
        if args.validate_only:
            validate_config(config)
            return

        # 解析selected_agents
        selected_agents = None
        if args.agents:
            selected_agents = [name.strip() for name in args.agents.split(',')]
            print(f"\n仅运行以下agents: {', '.join(selected_agents)}\n")

        # 并行模式提示
        if args.parallel:
            print(f"\n启用并行模式，max_workers={args.max_workers}")
            print("注意: 每个Agent将在独立进程中运行，避免Gurobi线程安全问题\n")

        # 运行实验
        run_experiment(
            config,
            selected_agents=selected_agents,
            limit=args.limit,
            parallel=args.parallel,
            max_workers=args.max_workers
        )

    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行失败: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
