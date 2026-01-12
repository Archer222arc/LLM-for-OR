"""
Benchmark runner for OR-Debug-Bench evaluation.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A4_Evaluation_Metrics.md

Key Components:
    - BenchmarkRunner: Run benchmarks and collect results
    - Parallel execution: Agent-level parallelism via ProcessPoolExecutor

Example:
    >>> from src.evaluation import BenchmarkRunner
    >>> runner = BenchmarkRunner()
    >>> results = runner.run_benchmark(envs, agent)
    >>> summary = runner.get_summary()

Parallel Example:
    >>> comparison = runner.compare_agents_parallel(
    ...     dataset_path="data/synthetic/debug_bench_v1/dataset.json",
    ...     agent_configs=[{"name": "gpt4", "type": "llm", ...}],
    ...     max_workers=4
    ... )
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import json

from src.environments import SolverDebugEnv, Action
from src.agents import BaseAgent
from .metrics import EpisodeResult, MetricsCalculator, BenchmarkConfig
from .episode_stats import EpisodeTracker


@dataclass
class BenchmarkProblem:
    """
    A benchmark problem instance.

    Attributes:
        problem_id: Unique identifier for the problem
        env: The environment instance
        ground_truth_fix: The known correct fix (if available)
        ground_truth_iis: The actual IIS constraints from solver
        metadata: Additional problem metadata
        original_objective: Original objective value (for OP calculation)
        original_constraints: Original constraint names (for FP calculation)
    """

    problem_id: str
    env: SolverDebugEnv
    ground_truth_fix: Optional[str] = None
    ground_truth_iis: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    original_objective: Optional[float] = None
    original_constraints: Optional[List[str]] = None


class BenchmarkRunner:
    """
    Run benchmarks and collect evaluation results.

    Orchestrates running agents on problem sets and collecting
    comprehensive evaluation metrics.

    Attributes:
        config: Benchmark configuration
        results: Collected episode results
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize BenchmarkRunner.

        Args:
            config: Benchmark configuration (defaults to BenchmarkConfig())
        """
        self.config = config or BenchmarkConfig()
        self.results: List[EpisodeResult] = []
        self._metrics_calculator = MetricsCalculator()

    def run_episode(
        self,
        env: SolverDebugEnv,
        agent: BaseAgent,
        problem_id: str = "",
        ground_truth_fix: Optional[str] = None,
        ground_truth_iis: Optional[List[str]] = None,
        original_objective: Optional[float] = None,
        original_constraints: Optional[List[str]] = None,
    ) -> EpisodeResult:
        """
        Run a single episode.

        Args:
            env: Environment instance
            agent: Agent to evaluate
            problem_id: Problem identifier
            ground_truth_fix: Known correct fix
            ground_truth_iis: Known IIS constraints for diagnosis accuracy
            original_objective: Original objective value (for OP calculation)
            original_constraints: Original constraint names (for FP calculation)

        Returns:
            EpisodeResult with episode data including token usage
        """
        tracker = EpisodeTracker(
            agent_name=agent.name,
            problem_id=problem_id,
            ground_truth_fix=ground_truth_fix,
            ground_truth_iis=ground_truth_iis or [],
            original_objective=original_objective,
            original_constraints=original_constraints or [],
        )

        # Reset environment and agent (agent.reset() also resets token stats)
        state, _ = env.reset()
        agent.reset()

        # Start timing for wall clock measurement
        start_time = time.time()

        done = False
        step = 0

        while not done and step < self.config.max_steps:
            # Get action from agent
            action = agent.act(state)

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Record step
            done = terminated or truncated
            tracker.record_step(state, action, reward, next_state, done)

            # Record for agent learning (optional)
            agent.record_step(state, action, reward, next_state, done)

            state = next_state
            step += 1

        # Measure elapsed time
        elapsed = time.time() - start_time

        # Determine success
        success = state.is_optimal()

        # Get token statistics from agent if available
        token_stats = None
        if hasattr(agent, 'get_token_stats'):
            token_stats = agent.get_token_stats()

        return tracker.finalize(
            success=success,
            token_stats=token_stats,
            elapsed_seconds=elapsed,
        )

    def run_benchmark(
        self,
        problems: List[BenchmarkProblem],
        agent: BaseAgent,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EpisodeResult]:
        """
        Run benchmark on a set of problems.

        Args:
            problems: List of benchmark problems
            agent: Agent to evaluate
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of episode results
        """
        results = []

        for i, problem in enumerate(problems):
            if progress_callback:
                progress_callback(i + 1, len(problems))

            if self.config.verbose:
                print(f"Running problem {i+1}/{len(problems)}: {problem.problem_id}")

            # Run n_episodes for each problem
            for ep in range(self.config.n_episodes):
                result = self.run_episode(
                    env=problem.env,
                    agent=agent,
                    problem_id=problem.problem_id,
                    ground_truth_fix=problem.ground_truth_fix,
                    ground_truth_iis=problem.ground_truth_iis,
                    original_objective=problem.original_objective,
                    original_constraints=problem.original_constraints,
                )
                results.append(result)

        self.results.extend(results)
        return results

    def compare_agents(
        self,
        problems: List[BenchmarkProblem],
        agents: List[BaseAgent],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple agents on the same problems.

        Args:
            problems: List of benchmark problems
            agents: List of agents to compare
            progress_callback: Optional callback(agent_name, current, total)

        Returns:
            Dictionary mapping agent names to their summaries
        """
        comparison = {}

        for agent in agents:
            if self.config.verbose:
                print(f"\nEvaluating agent: {agent.name}")

            def agent_progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback(agent.name, current, total)

            results = self.run_benchmark(
                problems=problems,
                agent=agent,
                progress_callback=agent_progress,
            )

            comparison[agent.name] = self._metrics_calculator.compute_summary(results)

        return comparison

    def get_summary(
        self,
        ground_truth_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Get summary of all collected results.

        Args:
            ground_truth_map: Optional ground truth for diagnosis accuracy

        Returns:
            Summary dictionary
        """
        return self._metrics_calculator.compute_summary(
            self.results, ground_truth_map
        )

    def format_summary(self) -> str:
        """
        Get formatted summary string.

        Returns:
            Formatted summary for display
        """
        summary = self.get_summary()
        return self._metrics_calculator.format_summary(summary)

    def format_comparison(
        self, comparison: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Format agent comparison as readable string.

        Args:
            comparison: Comparison dict from compare_agents

        Returns:
            Formatted comparison string
        """
        lines = [
            "=" * 60,
            "Agent Comparison Summary",
            "=" * 60,
            "",
            f"{'Agent':<20} {'Recovery':>10} {'Avg Steps':>12} {'Avg Reward':>12}",
            "-" * 60,
        ]

        for agent_name, metrics in comparison.items():
            lines.append(
                f"{agent_name:<20} "
                f"{metrics['recovery_rate']:>10.2%} "
                f"{metrics['avg_steps']:>12.2f} "
                f"{metrics['avg_reward']:>12.2f}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def clear_results(self) -> None:
        """Clear all collected results."""
        self.results = []

    def compare_agents_parallel(
        self,
        dataset_path: str,
        agent_configs: List[Dict[str, Any]],
        max_workers: int = 4,
        limit: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple agents in parallel using ProcessPoolExecutor.

        Each agent runs in a separate process, creating its own environments
        and Gurobi solver instances. This avoids Gurobi thread-safety issues.

        Args:
            dataset_path: Path to dataset JSON file
            agent_configs: List of agent configuration dicts
            max_workers: Maximum number of parallel workers
            limit: Limit number of problems (for testing)

        Returns:
            Dictionary mapping agent names to their summaries
        """
        print(f"\n{'='*60}")
        print(f"并行评估 {len(agent_configs)} 个Agent (max_workers={max_workers})")
        print(f"{'='*60}\n")

        # Prepare worker arguments
        worker_args_list = []
        for agent_config in agent_configs:
            worker_args = {
                'dataset_path': str(dataset_path),
                'agent_config': agent_config,
                'benchmark_config': {
                    'max_steps': self.config.max_steps,
                    'n_episodes': self.config.n_episodes,
                    'verbose': self.config.verbose,
                },
                'limit': limit,
            }
            worker_args_list.append(worker_args)

        # Run in parallel
        comparison = {}
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_evaluate_single_agent_worker, args): args['agent_config']['name']
                for args in worker_args_list
            }

            # Collect results as they complete
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    comparison[agent_name] = result
                    print(f"✓ {agent_name} 完成 - Recovery: {result['recovery_rate']:.1%}")
                except Exception as e:
                    print(f"✗ {agent_name} 失败: {e}")
                    comparison[agent_name] = {
                        'error': str(e),
                        'recovery_rate': 0.0,
                        'avg_steps': 0.0,
                        'avg_reward': 0.0,
                    }

        elapsed = time.time() - start_time
        print(f"\n并行评估完成，总用时: {elapsed:.1f}秒")

        return comparison


def _evaluate_single_agent_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for parallel agent evaluation.

    This function runs in a separate process and creates its own
    environments and agent instances. It must be a module-level
    function to be picklable.

    Args:
        args: Dictionary containing:
            - dataset_path: Path to dataset JSON
            - agent_config: Agent configuration dict
            - benchmark_config: Benchmark configuration dict
            - limit: Optional problem limit

    Returns:
        Metrics summary dictionary
    """
    # Import inside worker to avoid pickle issues
    from src.agents import (
        LLMAgent, HeuristicAgent, RandomAgent,
        GreedyDropAgent, DoNothingAgent
    )
    from src.solvers import GurobiSolver
    from src.environments import SolverDebugEnv

    dataset_path = args['dataset_path']
    agent_config = args['agent_config']
    benchmark_config = args['benchmark_config']
    limit = args.get('limit')

    agent_name = agent_config['name']

    # Load dataset and create problems
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    problems = []
    problem_list = dataset['problems'][:limit] if limit else dataset['problems']
    dataset_dir = Path(dataset_path).parent

    for p in problem_list:
        model_path = Path(p['model_file'])
        if not model_path.is_absolute():
            model_path = dataset_dir / model_path

        if not model_path.exists():
            continue

        solver = GurobiSolver.from_file(str(model_path))
        env = SolverDebugEnv(
            solver,
            problem_nl=p.get('problem_nl', ''),
            max_steps=50
        )

        problem = BenchmarkProblem(
            problem_id=p['problem_id'],
            env=env,
            ground_truth_fix=p.get('ground_truth_fix'),
            ground_truth_iis=p.get('iis_constraints', []),
            metadata={
                'error_type': p.get('error_type'),
                'error_description': p.get('error_description'),
            },
            original_objective=p.get('original_objective'),
            original_constraints=p.get('original_constraints', []),
        )
        problems.append(problem)

    # Create agent
    agent_type = agent_config['type']

    if agent_type == 'llm':
        provider = agent_config['provider']
        llm_params = {
            'model': agent_config['model'],
            'provider': provider,
            'temperature': agent_config.get('temperature', 0.0),
            'max_retries': agent_config.get('max_retries', 3),
            'name': agent_name
        }

        if provider == "azure_openai":
            llm_params['azure_endpoint'] = agent_config.get('azure_endpoint')
            llm_params['api_version'] = agent_config.get('api_version', '2024-10-21')
            llm_params['azure_deployment'] = agent_config.get('azure_deployment')

        agent = LLMAgent(**llm_params)

    elif agent_type == 'baseline':
        cls_name = agent_config['class']
        if cls_name == 'HeuristicAgent':
            agent = HeuristicAgent(name=agent_name)
        elif cls_name == 'RandomAgent':
            agent = RandomAgent(seed=agent_config.get('seed', 42), name=agent_name)
        elif cls_name == 'GreedyDropAgent':
            agent = GreedyDropAgent(name=agent_name)
        elif cls_name == 'DoNothingAgent':
            agent = DoNothingAgent(name=agent_name)
        else:
            raise ValueError(f"Unknown baseline agent: {cls_name}")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Create runner and evaluate
    config = BenchmarkConfig(
        max_steps=benchmark_config['max_steps'],
        n_episodes=benchmark_config['n_episodes'],
        verbose=False,  # Suppress verbose in workers
    )
    runner = BenchmarkRunner(config=config)

    results = runner.run_benchmark(problems=problems, agent=agent)

    # Compute summary
    metrics_calculator = MetricsCalculator()
    summary = metrics_calculator.compute_summary(results)

    return summary
