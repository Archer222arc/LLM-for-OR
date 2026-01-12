#!/usr/bin/env python3
"""
LLM Evaluation Script for OR-Debug-Bench.

Evaluates multiple LLM models on the OR-Debug-Bench dataset and generates
comprehensive evaluation reports with metrics including RR@k, OP, FP, DA.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/04_EVAL.md

Key Features:
    - SQLite database for concurrent-safe result storage
    - Incremental saving (each result saved immediately)
    - Resume/accumulation support (--resume)
    - Sample-level parallelism (--workers)
    - JSON import/export (--import-json, --export-json)

Usage:
    # Evaluate specific model with SQLite storage
    python scripts/evaluation/evaluate_llm.py --model gpt-5.2-chat --limit 200 --workers 4

    # Resume interrupted evaluation
    python scripts/evaluation/evaluate_llm.py --model gpt-5.2-chat --limit 200 --resume

    # Import existing JSON results to database
    python scripts/evaluation/evaluate_llm.py --import-json outputs/results.json

    # Export database to JSON
    python scripts/evaluation/evaluate_llm.py --export-json outputs/results.json

    # Monitor progress (external command)
    sqlite3 outputs/results.db "SELECT model_name, COUNT(*) FROM evaluation_results GROUP BY model_name"

Example:
    >>> python scripts/evaluation/evaluate_llm.py \\
    ...     --model o4-mini --limit 200 --workers 4 --db outputs/results.db
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents import LLMAgent, HeuristicAgent, RandomAgent, GreedyDropAgent, DoNothingAgent
from src.solvers import GurobiSolver
from src.environments import SolverDebugEnv
from src.evaluation import (
    BenchmarkRunner,
    BenchmarkProblem,
    BenchmarkConfig,
    MetricsCalculator,
    EpisodeResult,
    ResultDB,
)


def load_dataset(dataset_path: str, limit: Optional[int] = None) -> List[BenchmarkProblem]:
    """
    Load dataset and create BenchmarkProblem instances.

    Args:
        dataset_path: Path to dataset directory or JSON file
        limit: Optional limit on number of problems

    Returns:
        List of BenchmarkProblem instances
    """
    # Handle both directory and file paths
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        json_path = dataset_path / "dataset.json"
    else:
        json_path = dataset_path

    if not json_path.exists():
        raise FileNotFoundError(f"Dataset not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    problems = []
    problem_list = dataset.get('problems', [])
    if limit:
        problem_list = problem_list[:limit]

    dataset_dir = json_path.parent

    for p in problem_list:
        model_path = Path(p['model_file'])
        if not model_path.is_absolute():
            model_path = dataset_dir / model_path

        if not model_path.exists():
            print(f"Warning: Model file not found: {model_path}")
            continue

        try:
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
                    'difficulty': p.get('difficulty'),
                },
                original_objective=p.get('original_objective'),
                original_constraints=p.get('original_constraints', []),
            )
            problems.append(problem)
        except Exception as e:
            print(f"Warning: Failed to load problem {p.get('problem_id', 'unknown')}: {e}")
            continue

    print(f"Loaded {len(problems)} problems from {json_path}")
    return problems


def create_agent(config: Dict[str, Any]) -> Any:
    """
    Create agent from configuration dict.

    Args:
        config: Agent configuration dictionary

    Returns:
        Agent instance
    """
    agent_type = config.get('type', 'llm')
    name = config.get('name', 'agent')

    if agent_type == 'llm':
        return LLMAgent(
            model=config['model'],
            provider=config.get('provider', 'azure_openai'),
            temperature=config.get('temperature', 0.0),
            max_retries=config.get('max_retries', 3),
            name=name,
            use_local_config=config.get('use_local_config', True),
        )
    elif agent_type == 'baseline':
        cls_name = config.get('class', 'HeuristicAgent')
        if cls_name == 'HeuristicAgent':
            return HeuristicAgent(name=name)
        elif cls_name == 'RandomAgent':
            return RandomAgent(seed=config.get('seed', 42), name=name)
        elif cls_name == 'GreedyDropAgent':
            return GreedyDropAgent(name=name)
        elif cls_name == 'DoNothingAgent':
            return DoNothingAgent(name=name)
        else:
            raise ValueError(f"Unknown baseline agent: {cls_name}")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_agent(
    agent_config: Dict[str, Any],
    problems: List[BenchmarkProblem],
    max_steps: int = 20,
    verbose: bool = True,
    workers: int = 1,
    db: Optional[ResultDB] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single agent on the problem set.

    Args:
        agent_config: Agent configuration dict
        problems: List of problems to evaluate
        max_steps: Maximum steps per episode
        verbose: Print progress
        workers: Number of parallel workers (1 = sequential)
        db: Optional ResultDB for incremental saving

    Returns:
        Results dictionary with metrics
    """
    agent_name = agent_config.get('name', 'unknown')
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {agent_name} (workers={workers})")
        print(f"{'='*60}")

    # Thread-local storage for agents (each thread needs its own agent instance)
    thread_local = threading.local()

    def get_agent():
        """Get or create thread-local agent instance."""
        if not hasattr(thread_local, 'agent'):
            thread_local.agent = create_agent(agent_config)
        return thread_local.agent

    # Create shared runner config
    config = BenchmarkConfig(
        max_steps=max_steps,
        n_episodes=1,
        verbose=False,
    )

    # Progress counter for parallel execution
    progress_lock = threading.Lock()
    completed_count = [0]

    def run_single_problem(problem: BenchmarkProblem) -> EpisodeResult:
        """Evaluate a single problem (thread-safe)."""
        try:
            agent = get_agent()
            runner = BenchmarkRunner(config=config)
            result = runner.run_episode(
                env=problem.env,
                agent=agent,
                problem_id=problem.problem_id,
                ground_truth_fix=problem.ground_truth_fix,
                ground_truth_iis=problem.ground_truth_iis,
                original_objective=problem.original_objective,
                original_constraints=problem.original_constraints,
            )
            problem.env.reset()

            # Save result immediately to database (incremental saving)
            if db is not None:
                db.save_episode_result(agent_name, result)

            # Update progress
            with progress_lock:
                completed_count[0] += 1
                if verbose and completed_count[0] % 10 == 0:
                    print(f"  Progress: {completed_count[0]}/{len(problems)}")

            return result
        except Exception as e:
            print(f"  Error on problem {problem.problem_id}: {e}")
            error_result = EpisodeResult(
                success=False,
                final_status="ERROR",
                steps=0,
                total_reward=0.0,
                problem_id=problem.problem_id,
                agent_name=agent_name,
            )
            # Save error result too
            if db is not None:
                db.save_episode_result(agent_name, error_result)
            return error_result

    # Run evaluation
    start_time = time.time()
    results = []

    if workers <= 1:
        # Sequential execution (original behavior)
        for i, problem in enumerate(problems):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(problems)}")
            result = run_single_problem(problem)
            results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_single_problem, p): p for p in problems}
            for future in as_completed(futures):
                results.append(future.result())

    elapsed = time.time() - start_time

    # Compute metrics
    calculator = MetricsCalculator()
    summary = calculator.compute_summary(results)
    summary['agent_name'] = agent_name
    summary['elapsed_seconds'] = elapsed
    summary['agent_config'] = agent_config

    # Breakdown by error type
    error_types = set()
    for problem in problems:
        if problem.metadata:
            error_types.add(problem.metadata.get('error_type'))

    if error_types:
        summary['by_error_type'] = {}
        for error_type in error_types:
            if error_type is None:
                continue
            type_results = [
                r for r, p in zip(results, problems)
                if p.metadata and p.metadata.get('error_type') == error_type
            ]
            if type_results:
                summary['by_error_type'][error_type] = {
                    'n': len(type_results),
                    'recovery_rate': calculator.compute_recovery_rate(type_results),
                    'rr_at_5': calculator.compute_rr_at_k(type_results, 5),
                    'rr_at_10': calculator.compute_rr_at_k(type_results, 10),
                }

    # Breakdown by difficulty
    difficulties = set()
    for problem in problems:
        if problem.metadata:
            difficulties.add(problem.metadata.get('difficulty'))

    if difficulties:
        summary['by_difficulty'] = {}
        for diff in difficulties:
            if diff is None:
                continue
            diff_results = [
                r for r, p in zip(results, problems)
                if p.metadata and p.metadata.get('difficulty') == diff
            ]
            if diff_results:
                summary['by_difficulty'][diff] = {
                    'n': len(diff_results),
                    'recovery_rate': calculator.compute_recovery_rate(diff_results),
                    'rr_at_5': calculator.compute_rr_at_k(diff_results, 5),
                    'rr_at_10': calculator.compute_rr_at_k(diff_results, 10),
                }

    if verbose:
        print(f"\n  Results for {agent_name}:")
        print(f"    Recovery Rate: {summary['recovery_rate']:.2%}")
        print(f"    RR@5:  {summary.get('rr_at_5', 0):.2%}")
        print(f"    RR@10: {summary.get('rr_at_10', 0):.2%}")
        print(f"    RR@20: {summary.get('rr_at_20', 0):.2%}")
        print(f"    Avg Steps: {summary['avg_steps']:.2f}")
        print(f"    Time: {elapsed:.1f}s")

    return summary, results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation config from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_existing_file(output_path: str) -> Dict[str, Any]:
    """
    Load existing output file for multi-model result storage.

    Returns:
        Existing data dict or empty structure
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return {"models": {}}

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Ensure models key exists
        if "models" not in data:
            data["models"] = {}
        return data
    except Exception as e:
        print(f"Warning: Failed to load existing file: {e}")
        return {"models": {}}


def load_model_results(output_path: str, model_name: str) -> tuple:
    """
    Load existing results for a specific model from output file.

    Returns:
        Tuple of (completed_problem_ids: set, existing_episode_results: list)
    """
    data = load_existing_file(output_path)
    model_data = data.get("models", {}).get(model_name, {})

    if not model_data:
        return set(), []

    completed_ids = set()
    existing_results = []

    # Extract problem IDs from per_problem_results
    per_problem = model_data.get('per_problem_results', [])
    for pr in per_problem:
        problem_id = pr.get('problem_id')
        if problem_id:
            completed_ids.add(problem_id)
            # Reconstruct EpisodeResult
            existing_results.append(EpisodeResult(
                success=pr.get('success', False),
                final_status=pr.get('final_status', 'UNKNOWN'),
                steps=pr.get('steps', 0),
                total_reward=pr.get('total_reward', 0.0),
                problem_id=problem_id,
                agent_name=pr.get('agent_name', ''),
                diagnosed_constraints=pr.get('diagnosed_constraints', []),
                ground_truth_iis=pr.get('ground_truth_iis', []),
                original_objective=pr.get('original_objective'),
                recovered_objective=pr.get('recovered_objective'),
            ))

    print(f"Loaded {len(completed_ids)} existing results for {model_name}")
    return completed_ids, existing_results


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def import_json_to_db(json_path: str, db: ResultDB) -> int:
    """
    Import results from JSON file to database.

    Args:
        json_path: Path to JSON file with results
        db: ResultDB instance

    Returns:
        Total number of records imported
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_imported = 0

    for model_name, model_data in data.get('models', {}).items():
        per_problem = model_data.get('per_problem_results', [])
        imported_count = 0

        for pr in per_problem:
            result = EpisodeResult(
                problem_id=pr['problem_id'],
                success=pr['success'],
                final_status=pr['final_status'],
                steps=pr['steps'],
                total_reward=pr['total_reward'],
                agent_name=pr.get('agent_name', model_name),
                diagnosed_constraints=pr.get('diagnosed_constraints', []),
                ground_truth_iis=pr.get('ground_truth_iis', []),
                original_objective=pr.get('original_objective'),
                recovered_objective=pr.get('recovered_objective'),
            )
            if db.save_episode_result(model_name, result):
                imported_count += 1

        print(f"Imported {imported_count} results for {model_name}")
        total_imported += imported_count

    return total_imported


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print comparison table of all agents."""
    print("\n" + "=" * 100)
    print("Comparison Summary")
    print("=" * 100)

    # Check if any results have token data
    has_token_data = any(r.get('avg_tokens', 0) > 0 for r in results)

    # Header
    if has_token_data:
        print(f"{'Agent':<22} {'RR':<7} {'RR@5':<7} {'Steps':<7} {'Tokens':<9} {'Tok/Step':<9} {'Efficiency':<10}")
        print("-" * 100)
    else:
        print(f"{'Agent':<25} {'RR':<8} {'RR@5':<8} {'RR@10':<8} {'RR@20':<8} {'Steps':<8}")
        print("-" * 80)

    # Sort by recovery rate
    sorted_results = sorted(results, key=lambda x: x.get('recovery_rate', 0), reverse=True)

    for r in sorted_results:
        name = r.get('agent_name', 'unknown')[:21]
        rr = r.get('recovery_rate', 0)
        rr5 = r.get('rr_at_5', 0)
        rr10 = r.get('rr_at_10', 0)
        rr20 = r.get('rr_at_20', 0)
        steps = r.get('avg_steps', 0)

        if has_token_data:
            tokens = r.get('avg_tokens', 0)
            tok_per_step = r.get('tokens_per_step', 0)
            efficiency = r.get('token_efficiency', 0)
            print(f"{name:<22} {rr:>5.1%}  {rr5:>5.1%}  {steps:>5.1f}  {tokens:>8,.0f} {tok_per_step:>8,.1f}  {efficiency:>9.4f}")
        else:
            print(f"{name:<25} {rr:>6.1%}  {rr5:>6.1%}  {rr10:>6.1%}  {rr20:>6.1%}  {steps:>6.1f}")

    print("=" * 100 if has_token_data else "=" * 80)

    # Print token budget analysis if token data available
    if has_token_data:
        print("\nRecovery Rate @ Token Budget:")
        print("-" * 60)
        print(f"{'Agent':<22} {'RR@5k':<9} {'RR@10k':<9} {'RR@20k':<9} {'RR@50k':<9}")
        print("-" * 60)
        for r in sorted_results:
            name = r.get('agent_name', 'unknown')[:21]
            rr_5k = r.get('rr_at_5k_tokens', 0)
            rr_10k = r.get('rr_at_10k_tokens', 0)
            rr_20k = r.get('rr_at_20k_tokens', 0)
            rr_50k = r.get('rr_at_50k_tokens', 0)
            print(f"{name:<22} {rr_5k:>7.1%}  {rr_10k:>7.1%}  {rr_20k:>7.1%}  {rr_50k:>7.1%}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models on OR-Debug-Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to evaluation config YAML file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/benchmarks/or_debug_bench_full',
        help='Path to dataset directory or JSON file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Single model name to evaluate (overrides config)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='azure_openai',
        help='LLM provider (azure_openai, openai, anthropic)'
    )

    # Evaluation options
    parser.add_argument(
        '--max_steps',
        type=int,
        default=20,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of problems (for testing)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for sample evaluation (default: 1 = sequential)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing results, skip already evaluated problems'
    )

    # Database options
    parser.add_argument(
        '--db',
        type=str,
        default='outputs/results.db',
        help='SQLite database file for storing results (default: outputs/results.db)'
    )
    parser.add_argument(
        '--export-json',
        type=str,
        help='Export database to JSON file and exit'
    )
    parser.add_argument(
        '--import-json',
        type=str,
        help='Import results from JSON file to database and exit'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Initialize database
    db = ResultDB(args.db)
    print(f"Using database: {args.db}")

    # Handle export-json mode
    if args.export_json:
        db.export_json(args.export_json)
        db.close()
        return 0

    # Handle import-json mode
    if args.import_json:
        total = import_json_to_db(args.import_json, db)
        print(f"\nTotal imported: {total} records")
        db.close()
        return 0

    # Load problems
    problems = load_dataset(args.dataset, limit=args.limit)
    if not problems:
        print("Error: No problems loaded")
        return 1

    # Determine agents to evaluate
    agent_configs = []

    if args.model:
        # Single model mode
        agent_configs.append({
            'name': args.model,
            'type': 'llm',
            'model': args.model,
            'provider': args.provider,
        })
    elif args.config:
        # Config file mode
        config = load_config(args.config)
        agent_configs = config.get('agents', [])
    else:
        # Default: evaluate baselines
        agent_configs = [
            {'name': 'HeuristicAgent', 'type': 'baseline', 'class': 'HeuristicAgent'},
            {'name': 'GreedyDropAgent', 'type': 'baseline', 'class': 'GreedyDropAgent'},
            {'name': 'RandomAgent', 'type': 'baseline', 'class': 'RandomAgent'},
        ]

    # Run evaluation for each agent
    all_results = []

    for agent_config in agent_configs:
        model_name = agent_config.get('name', 'unknown')
        current_problems = list(problems)  # Copy for this agent

        # Resume logic: check database for completed problems
        existing_episode_results = []
        if args.resume:
            completed_ids = db.get_completed_problems(model_name)
            if completed_ids:
                original_count = len(current_problems)
                current_problems = [p for p in current_problems if p.problem_id not in completed_ids]
                print(f"Resume [{model_name}]: {len(completed_ids)} completed, {len(current_problems)} remaining (of {original_count})")
                if not current_problems:
                    print(f"  All problems already completed for {model_name}!")
                    # Load existing results for summary
                    existing_episode_results = db.get_model_results(model_name)
                    calculator = MetricsCalculator()
                    summary = calculator.compute_summary(existing_episode_results)
                    summary['agent_name'] = model_name
                    all_results.append(summary)
                    continue
                # Load existing results for merging
                existing_episode_results = db.get_model_results(model_name)

        # Run evaluation (results are saved incrementally to db)
        summary, episode_results = evaluate_agent(
            agent_config=agent_config,
            problems=current_problems,
            max_steps=args.max_steps,
            verbose=not args.quiet,
            workers=args.workers,
            db=db,
        )

        # Merge with existing results if resuming
        if existing_episode_results:
            all_episodes = existing_episode_results + episode_results
            # Recompute summary with all episodes
            calculator = MetricsCalculator()
            summary = calculator.compute_summary(all_episodes)
            summary['agent_name'] = model_name
            summary['agent_config'] = agent_config

        # Update summary in database
        db.update_summary(
            model_name, summary,
            agent_config=agent_config,
            elapsed_seconds=summary.get('elapsed_seconds')
        )

        all_results.append(summary)

        # Re-load problems to reset environments (for multi-agent configs)
        if len(agent_configs) > 1:
            problems = load_dataset(args.dataset, limit=args.limit)

    # Print comparison
    if not args.quiet:
        print_comparison_table(all_results)

    # Print database stats
    stats = db.get_stats()
    print(f"\nDatabase stats ({args.db}):")
    for model, s in stats.items():
        print(f"  {model}: {s['count']} problems, RR={s['recovery_rate']:.1%}")

    # Also save to JSON if --output specified
    if args.output:
        db.export_json(args.output)

    db.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
