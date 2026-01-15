#!/usr/bin/env python3
"""
Create a held-out test benchmark by excluding training samples.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/04_EVAL.md

Key Features:
    - Exclude training problem IDs from evaluation benchmark
    - Preserve models directory via symlink
    - Recalculate statistics for held-out subset
    - Support optional size limits

Usage:
    python scripts/evaluation/create_holdout_test.py \
        --benchmark data/benchmarks/or_debug_bench_full \
        --exclude data/training/grpo_training_ids.txt \
        --output data/benchmarks/or_debug_bench_holdout

Example:
    >>> # Create held-out set excluding 300 training samples
    >>> python scripts/evaluation/create_holdout_test.py \\
    ...     --benchmark data/benchmarks/or_debug_bench_full \\
    ...     --exclude data/training/grpo_training_ids.txt \\
    ...     --output data/benchmarks/or_debug_bench_holdout
    Loaded 300 IDs to exclude
    Full benchmark has 4462 problems
    Held-out set has 4162 problems
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create held-out test benchmark")
    parser.add_argument("--benchmark", required=True, help="Path to full benchmark")
    parser.add_argument("--exclude", required=True, help="File with problem IDs to exclude")
    parser.add_argument("--output", required=True, help="Output benchmark directory")
    parser.add_argument("--limit", type=int, help="Optional limit on output size")
    args = parser.parse_args()

    # Load exclusion list
    with open(args.exclude, 'r') as f:
        exclude_ids = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(exclude_ids)} IDs to exclude")

    # Load full benchmark
    benchmark_dir = Path(args.benchmark)
    with open(benchmark_dir / "dataset.json", 'r') as f:
        data = json.load(f)

    total_problems = len(data['problems'])
    print(f"Full benchmark has {total_problems} problems")

    # Filter out training problems
    holdout_problems = [
        p for p in data['problems']
        if p['problem_id'] not in exclude_ids
    ]
    print(f"Held-out set has {len(holdout_problems)} problems")

    # Apply limit if specified
    if args.limit and len(holdout_problems) > args.limit:
        holdout_problems = holdout_problems[:args.limit]
        print(f"Limited to {len(holdout_problems)} problems")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create symlink to models directory
    models_link = output_dir / "models"
    if models_link.exists():
        if models_link.is_symlink():
            models_link.unlink()
        else:
            raise RuntimeError(f"{models_link} exists and is not a symlink")

    source_models = (benchmark_dir / "models").resolve()
    models_link.symlink_to(source_models)
    print(f"Created symlink: {models_link} -> {source_models}")

    # Update metadata
    data['dataset_name'] = 'or_debug_bench_holdout'
    data['description'] = f"Held-out test set (excluded {len(exclude_ids)} training samples)"
    data['num_problems'] = len(holdout_problems)
    data['problems'] = holdout_problems

    # Recalculate statistics
    success_by_type = {}
    difficulty_dist = {}
    for p in holdout_problems:
        etype = p.get('error_type', 'unknown')
        diff = p.get('difficulty', 'unknown')
        success_by_type[etype] = success_by_type.get(etype, 0) + 1
        difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

    data['success_by_type'] = success_by_type
    data['difficulty_distribution'] = difficulty_dist

    # Save dataset
    output_json = output_dir / "dataset.json"
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved held-out benchmark to {output_json}")

    # Print summary
    print(f"\n=== Held-Out Benchmark Summary ===")
    print(f"Total problems: {len(holdout_problems)}")
    print(f"By error type: {success_by_type}")
    print(f"By difficulty: {difficulty_dist}")


if __name__ == "__main__":
    main()
