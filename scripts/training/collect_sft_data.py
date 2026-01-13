#!/usr/bin/env python3
"""
SFT Data Collection Script for OR-Debug-Bench.

Collects successful debugging trajectories from strong models (teacher models)
for supervised fine-tuning of smaller models (e.g., Qwen3-8B).

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/05_TRAINING.md

Key Components:
    - collect_trajectory: Run agent on problem, extract successful trajectory
    - format_state_for_sft: Convert DebugState to SFT input format
    - format_action_for_sft: Convert Action to SFT output with <think> tags
    - save_result_incremental: Thread-safe JSON append with file locking

Features:
    - Parallel collection with --workers N (ThreadPoolExecutor)
    - Incremental by default (auto-resume if output file exists)
    - Use --overwrite to start fresh
    - Each result saved immediately (fcntl file locking)

Output Format:
    {
        "instruction": "Debug the infeasible optimization model...",
        "input": "## Problem\\nID: ...\\n## IIS\\n...",
        "output": "<think>\\nStep N: reasoning...\\n</think>\\n\\nAction: ...",
        "metadata": {"problem_id": ..., "error_type": ..., "steps": ...}
    }

Usage:
    # Collect from HeuristicAgent (baseline, fast)
    python scripts/training/collect_sft_data.py \\
        --agent heuristic \\
        --dataset data/benchmarks/or_debug_bench_full \\
        --output data/training/sft_heuristic.json

    # Collect from LLM with parallelism (auto-resumes if file exists)
    python scripts/training/collect_sft_data.py \\
        --agent llm \\
        --model gpt-5.2-chat \\
        --dataset data/benchmarks/or_debug_bench_full \\
        --output data/training/sft_gpt52chat.json \\
        --max_steps 5 \\
        --workers 4

    # Overwrite existing file
    python scripts/training/collect_sft_data.py \\
        --agent llm --model gpt-5.2-chat ... --overwrite
"""

import argparse
import fcntl
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents import LLMAgent, HeuristicAgent, GreedyDropAgent
from src.solvers import GurobiSolver
from src.environments import SolverDebugEnv, DebugState, Action
from src.evaluation import BenchmarkProblem, EpisodeResult


def load_dataset(dataset_path: str, limit: Optional[int] = None, offset: int = 0) -> Tuple[List[Dict[str, Any]], Path]:
    """Load dataset and return problem metadata."""
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        json_path = dataset_path / "dataset.json"
    else:
        json_path = dataset_path

    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    problems = dataset.get('problems', [])

    # Apply offset first
    if offset > 0:
        problems = problems[offset:]

    # Then apply limit
    if limit:
        problems = problems[:limit]

    return problems, json_path.parent


def create_agent(agent_type: str, model: Optional[str] = None) -> Any:
    """Create agent based on type."""
    if agent_type == 'heuristic':
        return HeuristicAgent(name='SFT-Teacher-Heuristic')
    elif agent_type == 'greedy':
        return GreedyDropAgent(name='SFT-Teacher-Greedy')
    elif agent_type == 'llm':
        if not model:
            raise ValueError("LLM agent requires --model parameter")
        return LLMAgent(
            model=model,
            provider='azure_openai',
            temperature=0.0,
            name=f'SFT-Teacher-{model}',
            use_local_config=True,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def format_state_for_sft(state: DebugState, problem_meta: Dict[str, Any]) -> str:
    """Format state as input for SFT."""
    lines = [
        f"## Problem",
        f"ID: {problem_meta.get('problem_id', 'unknown')}",
        f"Type: {problem_meta.get('error_type', 'unknown')}",
        "",
        f"## Current State",
        f"Status: {state.solver_status}",
    ]

    if state.iis_constraints or state.iis_bounds:
        lines.append("")
        lines.append("## IIS (Irreducible Infeasible Subsystem)")
        if state.iis_constraints:
            lines.append(f"Conflicting Constraints: {state.iis_constraints}")
        if state.iis_bounds:
            lines.append(f"Conflicting Bounds: {state.iis_bounds}")

    lines.append("")
    lines.append("## Model Structure")
    lines.append(f"Constraints: {len(state.constraint_names)}")
    lines.append(f"Variables: {len(state.variable_names)}")

    if state.constraint_names and len(state.constraint_names) <= 15:
        lines.append(f"Constraint Names: {state.constraint_names}")

    return "\n".join(lines)


def format_action_for_sft(
    action: Action,
    state: DebugState,
    next_state: DebugState,
    step: int,
) -> str:
    """Format action and reasoning as output for SFT."""
    reasoning_lines = []

    if action.action_type.value == 'get_iis':
        reasoning_lines.extend([
            f"Step {step}: Need to identify the source of infeasibility.",
            "Using GET_IIS to compute the Irreducible Infeasible Subsystem.",
        ])
    elif action.action_type.value == 'drop_constraint':
        reasoning_lines.extend([
            f"Step {step}: The IIS contains {len(state.iis_constraints)} constraints.",
            f"Constraint '{action.target}' is identified as problematic.",
            "Dropping this constraint should resolve the conflict.",
        ])
    elif action.action_type.value == 'relax_constraint':
        reasoning_lines.extend([
            f"Step {step}: Constraint '{action.target}' is too tight.",
            f"Relaxing by {action.value} to allow feasibility.",
        ])
    elif action.action_type.value == 'submit':
        reasoning_lines.extend([
            f"Step {step}: Model is now OPTIMAL.",
            "Submitting the repaired solution.",
        ])
    else:
        reasoning_lines.append(f"Step {step}: Taking action {action.action_type.value}")

    reasoning = "\n".join(reasoning_lines)

    action_str = action.action_type.value.upper()
    if action.target:
        if action.value is not None:
            action_str = f"{action_str}({action.target}, {action.value})"
        else:
            action_str = f"{action_str}({action.target})"

    output = f"<think>\n{reasoning}\n</think>\n\nAction: {action_str}"
    return output


def collect_trajectory(
    problem_meta: Dict[str, Any],
    dataset_dir: Path,
    agent: Any,
    max_steps: int = 5,
) -> Optional[Dict[str, Any]]:
    """Collect a single successful trajectory."""
    model_path = Path(problem_meta['model_file'])
    if not model_path.is_absolute():
        model_path = dataset_dir / model_path

    if not model_path.exists():
        return None

    try:
        solver = GurobiSolver.from_file(str(model_path))
        env = SolverDebugEnv(
            solver,
            problem_nl=problem_meta.get('problem_nl', ''),
            max_steps=max_steps + 5
        )

        state, _ = env.reset()
        agent.reset()

        trajectory = []
        step = 0

        while step < max_steps:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            trajectory.append({
                'step': step + 1,
                'state': state,
                'action': action,
                'next_state': next_state,
            })

            if next_state.is_optimal():
                break

            state = next_state
            step += 1

        if not trajectory or not trajectory[-1]['next_state'].is_optimal():
            return None

        if len(trajectory) > max_steps:
            return None

        final_step = trajectory[-1]
        input_text = format_state_for_sft(final_step['state'], problem_meta)
        output_text = format_action_for_sft(
            final_step['action'],
            final_step['state'],
            final_step['next_state'],
            final_step['step'],
        )

        return {
            'instruction': "Debug the infeasible optimization model and provide the action to fix it.",
            'input': input_text,
            'output': output_text,
            'metadata': {
                'problem_id': problem_meta['problem_id'],
                'error_type': problem_meta.get('error_type'),
                'difficulty': problem_meta.get('difficulty'),
                'steps': len(trajectory),
                'agent': agent.name,
            }
        }

    except Exception as e:
        # Silently fail for individual problems
        return None


def init_output_file(output_path: Path, metadata: Dict[str, Any]):
    """Initialize output JSON file with metadata."""
    output = {
        'metadata': metadata,
        'data': [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def save_result_incremental(result: Dict[str, Any], output_path: Path):
    """Thread-safe incremental save to JSON file."""
    with open(output_path, 'r+', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
            data['data'].append(result)
            # Update counts in metadata
            data['metadata']['success_count'] = len(data['data'])
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_completed_ids(output_path: Path) -> set:
    """Load already completed problem IDs from output file."""
    if not output_path.exists():
        return set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {d['metadata']['problem_id'] for d in data.get('data', [])}
    except (json.JSONDecodeError, KeyError):
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Collect SFT training data from teacher models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--agent', type=str, default='heuristic', choices=['heuristic', 'greedy', 'llm'])
    parser.add_argument('--model', type=str, help='LLM model name (required if agent=llm)')
    parser.add_argument('--max_steps', type=int, default=5, help='Maximum steps for successful trajectory')
    parser.add_argument('--limit', type=int, help='Limit number of problems')
    parser.add_argument('--offset', type=int, default=0, help='Skip first N problems')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file (default: incremental)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')

    args = parser.parse_args()
    output_path = Path(args.output)

    # Load dataset
    problems, dataset_dir = load_dataset(args.dataset, limit=args.limit, offset=args.offset)
    total_problems = len(problems)
    print(f"Loaded {total_problems} problems from {args.dataset}")

    # Incremental collection by default (resume if file exists)
    completed_ids = set()
    if output_path.exists() and not args.overwrite:
        # Auto-resume: load completed IDs and skip them
        completed_ids = load_completed_ids(output_path)
        if completed_ids:
            original_count = len(problems)
            problems = [p for p in problems if p['problem_id'] not in completed_ids]
            print(f"Auto-resume: {len(completed_ids)} completed, {len(problems)} remaining (of {original_count})")
            if not problems:
                print("All problems already completed!")
                return 0
        else:
            print(f"Output file exists but empty, continuing collection")
    else:
        # Initialize new output file (overwrite if --overwrite specified)
        if output_path.exists() and args.overwrite:
            print(f"Overwriting existing file: {output_path}")
        init_output_file(output_path, {
            'created': datetime.now().isoformat(),
            'dataset': args.dataset,
            'agent': args.agent,
            'model': args.model,
            'max_steps': args.max_steps,
            'total_problems': total_problems,
            'success_count': 0,
            'fail_count': 0,
        })

    print(f"Using {args.workers} worker(s)")

    # Thread-local storage for agents
    thread_local = threading.local()

    def get_agent():
        if not hasattr(thread_local, 'agent'):
            thread_local.agent = create_agent(args.agent, args.model)
        return thread_local.agent

    # Progress tracking
    progress_lock = threading.Lock()
    success_count = [0]
    fail_count = [0]

    def worker_task(problem_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Worker function for parallel collection."""
        agent = get_agent()
        result = collect_trajectory(problem_meta, dataset_dir, agent, max_steps=args.max_steps)

        with progress_lock:
            if result:
                success_count[0] += 1
                save_result_incremental(result, output_path)
            else:
                fail_count[0] += 1

            total = success_count[0] + fail_count[0]
            if not args.quiet and total % 20 == 0:
                print(f"Progress: {total}/{len(problems)} (success={success_count[0]}, fail={fail_count[0]})")

        return result

    # Run collection
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker_task, p) for p in problems]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    if not args.quiet:
                        print(f"Worker error: {e}")
    else:
        agent = create_agent(args.agent, args.model)
        print(f"Using agent: {agent.name}")

        for i, problem_meta in enumerate(problems):
            result = collect_trajectory(problem_meta, dataset_dir, agent, max_steps=args.max_steps)

            if result:
                success_count[0] += 1
                save_result_incremental(result, output_path)
            else:
                fail_count[0] += 1

            if not args.quiet and (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(problems)} (success={success_count[0]}, fail={fail_count[0]})")

    # Final summary
    print(f"\n{'='*60}")
    print(f"SFT Data Collection Complete")
    print(f"{'='*60}")
    print(f"Problems processed: {len(problems)}")
    print(f"Successful:         {success_count[0]} ({success_count[0]/len(problems)*100:.1f}%)" if problems else "N/A")
    print(f"Failed:             {fail_count[0]}")
    print(f"Output saved to:    {output_path}")
    if completed_ids:
        print(f"Total in file:      {len(completed_ids) + success_count[0]}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
