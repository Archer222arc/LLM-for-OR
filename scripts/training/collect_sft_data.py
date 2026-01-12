#!/usr/bin/env python3
"""
SFT Data Collection Script for OR-Debug-Bench.

Collects successful debugging trajectories from strong models (teacher models)
for supervised fine-tuning of smaller models.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/05_TRAINING.md

Usage:
    # Collect from HeuristicAgent (as baseline)
    python scripts/training/collect_sft_data.py \
        --agent heuristic \
        --dataset data/benchmarks/or_debug_bench_full \
        --output data/training/sft_heuristic.json

    # Collect from LLM (requires API)
    python scripts/training/collect_sft_data.py \
        --model gpt-4.1 \
        --dataset data/benchmarks/or_debug_bench_full \
        --output data/training/sft_gpt4.json \
        --max_steps 5

Example SFT data format:
    {
        "instruction": "Debug the infeasible optimization model.",
        "input": "Model: [code]\\nStatus: INFEASIBLE\\nIIS: [c1, c2]",
        "output": "<think>\\n1. Analysis...\\n</think>\\n\\nAction: DROP_CONSTRAINT c1"
    }
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents import LLMAgent, HeuristicAgent, GreedyDropAgent
from src.solvers import GurobiSolver
from src.environments import SolverDebugEnv, DebugState, Action
from src.evaluation import BenchmarkProblem, EpisodeResult


def load_dataset(dataset_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load dataset and return problem metadata."""
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        json_path = dataset_path / "dataset.json"
    else:
        json_path = dataset_path

    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    problems = dataset.get('problems', [])
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
    """
    Format state as input for SFT.

    Args:
        state: Current environment state
        problem_meta: Problem metadata from dataset

    Returns:
        Formatted input string
    """
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
    """
    Format action and reasoning as output for SFT.

    Args:
        action: Action taken
        state: State before action
        next_state: State after action
        step: Step number

    Returns:
        Formatted output string with reasoning
    """
    # Generate reasoning based on action
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

    # Format output
    reasoning = "\n".join(reasoning_lines)

    # Format action string
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
    """
    Collect a single successful trajectory.

    Args:
        problem_meta: Problem metadata
        dataset_dir: Dataset directory path
        agent: Agent to use
        max_steps: Maximum steps for success

    Returns:
        SFT data dict if successful, None otherwise
    """
    # Load model
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
            max_steps=max_steps + 5  # Allow some buffer
        )

        # Reset
        state, _ = env.reset()
        agent.reset()

        # Collect trajectory
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
                # Success!
                break

            state = next_state
            step += 1

        # Check if successful within max_steps
        if not trajectory or not trajectory[-1]['next_state'].is_optimal():
            return None

        if len(trajectory) > max_steps:
            return None

        # Format as SFT data
        # For now, we'll format the final action that led to success
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
        print(f"Error processing {problem_meta.get('problem_id')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect SFT training data from teacher models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--agent',
        type=str,
        default='heuristic',
        choices=['heuristic', 'greedy', 'llm'],
        help='Agent type to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='LLM model name (required if agent=llm)'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=5,
        help='Maximum steps for successful trajectory'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of problems'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output'
    )

    args = parser.parse_args()

    # Load dataset
    problems, dataset_dir = load_dataset(args.dataset, limit=args.limit)
    print(f"Loaded {len(problems)} problems from {args.dataset}")

    # Create agent
    agent = create_agent(args.agent, args.model)
    print(f"Using agent: {agent.name}")

    # Collect trajectories
    sft_data = []
    success_count = 0
    fail_count = 0

    for i, problem_meta in enumerate(problems):
        if not args.quiet and (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(problems)} (success={success_count}, fail={fail_count})")

        result = collect_trajectory(
            problem_meta,
            dataset_dir,
            agent,
            max_steps=args.max_steps,
        )

        if result:
            sft_data.append(result)
            success_count += 1
        else:
            fail_count += 1

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'dataset': args.dataset,
            'agent': args.agent,
            'model': args.model,
            'max_steps': args.max_steps,
            'total_problems': len(problems),
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': success_count / len(problems) if problems else 0,
        },
        'data': sft_data,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SFT Data Collection Complete")
    print(f"{'='*60}")
    print(f"Total problems:  {len(problems)}")
    print(f"Successful:      {success_count} ({success_count/len(problems)*100:.1f}%)")
    print(f"Failed:          {fail_count}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
