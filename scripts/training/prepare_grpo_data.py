#!/usr/bin/env python3
"""
Prepare GRPO training dataset from SFT data or benchmark.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/05_TRAINING.md

Key Features:
    - Convert SFT chat format to GRPO prompt format
    - Direct loading from benchmark with stratified sampling
    - Support for n_prompts to control dataset size
    - Deduplication by problem_id

Usage:
    # From SFT data
    python scripts/training/prepare_grpo_data.py \
        --input data/training/sft_qwen_chat.train.jsonl \
        --output data/training/grpo_prompts.jsonl

    # From benchmark with stratified sampling (300 prompts)
    python scripts/training/prepare_grpo_data.py \
        --benchmark data/benchmarks/or_debug_bench_full \
        --output data/training/grpo_prompts_v2.jsonl \
        --n_prompts 300
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict


def extract_problem_metadata(user_content: str) -> Dict[str, Any]:
    """
    Extract problem metadata from user message content.

    Args:
        user_content: The user message content containing problem details

    Returns:
        Dict with problem_id, problem_type, iis_constraints, constraint_names
    """
    metadata = {
        "problem_id": None,
        "problem_type": None,
        "iis_constraints": [],
        "iis_bounds": [],
        "constraint_names": [],
        "model_file": None,
    }

    # Extract Problem ID
    id_match = re.search(r"ID:\s*(\S+)", user_content)
    if id_match:
        metadata["problem_id"] = id_match.group(1)

    # Extract Type
    type_match = re.search(r"Type:\s*(\w+)", user_content)
    if type_match:
        metadata["problem_type"] = type_match.group(1)

    # Extract IIS Constraints
    iis_match = re.search(r"Conflicting Constraints:\s*\[([^\]]*)\]", user_content)
    if iis_match:
        constraints_str = iis_match.group(1)
        constraints = re.findall(r"'([^']+)'", constraints_str)
        metadata["iis_constraints"] = constraints

    # Extract IIS Bounds
    bounds_match = re.search(r"Conflicting Bounds:\s*\[([^\]]*)\]", user_content)
    if bounds_match:
        bounds_str = bounds_match.group(1)
        bounds = re.findall(r"'([^']+)'", bounds_str)
        metadata["iis_bounds"] = bounds

    # Extract Constraint Names
    names_match = re.search(r"Constraint Names:\s*\[([^\]]*)\]", user_content)
    if names_match:
        names_str = names_match.group(1)
        names = re.findall(r"'([^']+)'", names_str)
        metadata["constraint_names"] = names

    # Construct model file path (if problem_id is available)
    if metadata["problem_id"]:
        metadata["model_file"] = f"data/benchmarks/models/{metadata['problem_id']}.mps"

    return metadata


def convert_sft_to_grpo(sft_record: Dict) -> Optional[Dict]:
    """
    Convert a single SFT record to GRPO format.

    Args:
        sft_record: SFT record with 'messages' field

    Returns:
        GRPO record with 'prompt' (as chat messages) and metadata fields
    """
    messages = sft_record.get("messages", [])

    if len(messages) < 2:
        return None

    # Find system and user messages
    system_content = ""
    user_content = ""

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            user_content = msg["content"]

    if not user_content:
        return None

    # Build prompt as list of messages (chat format for TRL GRPOTrainer)
    # TRL will apply the chat template automatically when prompt is a list
    prompt_messages = []
    if system_content:
        prompt_messages.append({"role": "system", "content": system_content})
    prompt_messages.append({"role": "user", "content": user_content})

    # Extract metadata
    metadata = extract_problem_metadata(user_content)

    # Create GRPO record with prompt as list of messages
    grpo_record = {
        "prompt": prompt_messages,
        "problem_id": metadata["problem_id"],
        "problem_type": metadata["problem_type"],
        "iis_constraints": json.dumps(metadata["iis_constraints"]),
        "iis_bounds": json.dumps(metadata["iis_bounds"]),
        "constraint_names": json.dumps(metadata["constraint_names"]),
        "model_file": metadata["model_file"],
    }

    return grpo_record


def load_benchmark(benchmark_path: Path) -> List[Dict]:
    """
    Load problems from benchmark directory.

    Args:
        benchmark_path: Path to benchmark directory containing dataset.json

    Returns:
        List of problem dictionaries
    """
    dataset_file = benchmark_path / "dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {dataset_file}")

    with open(dataset_file, "r") as f:
        data = json.load(f)

    return data.get("problems", [])


def stratified_sample(
    problems: List[Dict],
    n_prompts: int,
    seed: int = 2026
) -> List[Dict]:
    """
    Stratified sampling by error_type and difficulty.

    Args:
        problems: List of all problems
        n_prompts: Target number of prompts to select
        seed: Random seed for reproducibility

    Returns:
        Selected subset of problems
    """
    random.seed(seed)

    # Group problems by error_type and difficulty
    groups = defaultdict(list)
    for p in problems:
        error_type = p.get("error_type", "unknown")
        difficulty = p.get("difficulty", "medium")
        groups[(error_type, difficulty)].append(p)

    # Calculate samples per group (proportional)
    total_groups = len(groups)
    if total_groups == 0:
        return []

    # Calculate base allocation per group
    base_per_group = n_prompts // total_groups
    remaining = n_prompts % total_groups

    selected = []
    group_keys = sorted(groups.keys())

    for i, key in enumerate(group_keys):
        group_problems = groups[key]
        # Allocate extra samples to first 'remaining' groups
        n_select = base_per_group + (1 if i < remaining else 0)
        n_select = min(n_select, len(group_problems))

        if n_select > 0:
            sampled = random.sample(group_problems, n_select)
            selected.extend(sampled)

    # If we still need more (due to small groups), sample from all
    if len(selected) < n_prompts:
        remaining_problems = [p for p in problems if p not in selected]
        extra_needed = n_prompts - len(selected)
        if remaining_problems:
            extra = random.sample(
                remaining_problems,
                min(extra_needed, len(remaining_problems))
            )
            selected.extend(extra)

    random.shuffle(selected)
    return selected[:n_prompts]


def convert_benchmark_to_grpo(
    problem: Dict,
    benchmark_path: Path,
    system_prompt: str
) -> Dict:
    """
    Convert a benchmark problem to GRPO format.

    Args:
        problem: Problem dictionary from benchmark
        benchmark_path: Path to benchmark directory (for model file path)
        system_prompt: System prompt to use

    Returns:
        GRPO record with prompt and metadata
    """
    # Build user message content
    user_lines = [
        f"## Problem Information",
        f"- ID: {problem['problem_id']}",
        f"- Type: {problem.get('error_type', 'unknown')}",
        f"- Status: {problem.get('initial_status', 'INFEASIBLE')}",
        f"",
        f"## Problem Description",
        f"{problem.get('problem_nl', 'No description available.')}",
        f"",
        f"## Model Structure",
        f"- Variables: {problem.get('n_variables', 'unknown')}",
        f"- Constraints: {problem.get('n_constraints', 'unknown')}",
    ]

    # Add IIS information if available
    iis_constraints = problem.get("iis_constraints", [])
    iis_bounds = problem.get("iis_bounds", [])

    if iis_constraints or iis_bounds:
        user_lines.append("")
        user_lines.append("## IIS (Irreducible Infeasible Subsystem)")
        if iis_constraints:
            user_lines.append(f"- Conflicting Constraints: {iis_constraints}")
        if iis_bounds:
            user_lines.append(f"- Conflicting Bounds: {iis_bounds}")

    user_lines.append("")
    user_lines.append("What action should be taken to fix this infeasible model?")

    user_content = "\n".join(user_lines)

    # Build prompt as list of messages
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # Construct model file path
    model_file = str(benchmark_path / problem.get("model_file", ""))

    # Create GRPO record
    grpo_record = {
        "prompt": prompt_messages,
        "problem_id": problem["problem_id"],
        "problem_type": problem.get("error_type"),
        "iis_constraints": json.dumps(iis_constraints),
        "iis_bounds": json.dumps(iis_bounds),
        "constraint_names": json.dumps([]),  # Not in benchmark format
        "model_file": model_file,
        "difficulty": problem.get("difficulty"),
        "ground_truth_fix": problem.get("ground_truth_fix"),
        "target_name": problem.get("target_name"),
    }

    return grpo_record


def get_system_prompt() -> str:
    """Get the system prompt for GRPO training."""
    # Import from prompts module if available, otherwise use default
    try:
        from src.agents.prompts import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    except ImportError:
        return """You are an expert Operations Research debugger. Your task is to diagnose and repair infeasible optimization models.

## Your Goal
Transform an INFEASIBLE model into an OPTIMAL one by identifying and fixing constraint conflicts.

## Available Actions
- GET_IIS: Compute the Irreducible Infeasible Subsystem
- DROP_CONSTRAINT(constraint): Remove a constraint from the model
- RELAX_CONSTRAINT(constraint, epsilon): Relax constraint bounds by epsilon
- SUBMIT: Submit the current model as your solution

## Response Format
Respond with a valid JSON object:
```json
{
    "reasoning": "Brief explanation of your decision",
    "action": "ACTION_NAME",
    "target": "constraint_name",
    "value": 0.0
}
```"""


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GRPO dataset from SFT data or benchmark"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input SFT data file (mutually exclusive with --benchmark)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Path to benchmark directory (e.g., data/benchmarks/or_debug_bench_full)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/grpo_prompts.jsonl",
        help="Output GRPO prompts file"
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=300,
        help="Number of prompts to generate (default: 300, only for benchmark mode)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for stratified sampling (default: 2026)"
    )
    args = parser.parse_args()

    # Validate arguments
    if args.input and args.benchmark:
        print("Error: --input and --benchmark are mutually exclusive")
        return
    if not args.input and not args.benchmark:
        # Default to SFT mode with default input
        args.input = "data/training/sft_qwen_chat.train.jsonl"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        # Benchmark mode: load from benchmark with stratified sampling
        benchmark_path = Path(args.benchmark)
        print(f"Loading benchmark from: {benchmark_path}")

        problems = load_benchmark(benchmark_path)
        print(f"  Total problems in benchmark: {len(problems)}")

        # Stratified sampling
        selected = stratified_sample(problems, args.n_prompts, args.seed)
        print(f"  Selected {len(selected)} problems (stratified sampling)")

        # Show distribution
        type_counts = defaultdict(int)
        diff_counts = defaultdict(int)
        for p in selected:
            type_counts[p.get("error_type", "unknown")] += 1
            diff_counts[p.get("difficulty", "unknown")] += 1
        print(f"  By error_type: {dict(type_counts)}")
        print(f"  By difficulty: {dict(diff_counts)}")

        # Convert to GRPO format
        system_prompt = get_system_prompt()
        converted = 0

        with open(output_path, "w") as f_out:
            for problem in selected:
                grpo_record = convert_benchmark_to_grpo(
                    problem, benchmark_path, system_prompt
                )
                f_out.write(json.dumps(grpo_record) + "\n")
                converted += 1

        print(f"\nConversion complete!")
        print(f"  Converted: {converted} problems")
        print(f"  Output: {output_path}")

    else:
        # SFT mode: convert from existing SFT data
        input_path = Path(args.input)
        print(f"Converting SFT data from: {input_path}")

        converted = 0
        skipped = 0
        seen_problems = set()

        with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
            for line in f_in:
                if not line.strip():
                    continue

                try:
                    sft_record = json.loads(line)
                    grpo_record = convert_sft_to_grpo(sft_record)

                    if grpo_record and grpo_record["problem_id"]:
                        if grpo_record["problem_id"] not in seen_problems:
                            seen_problems.add(grpo_record["problem_id"])
                            f_out.write(json.dumps(grpo_record) + "\n")
                            converted += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    skipped += 1

        print(f"\nConversion complete!")
        print(f"  Converted: {converted} unique problems")
        print(f"  Skipped: {skipped} (duplicates or invalid)")
        print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
