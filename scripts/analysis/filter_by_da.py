#!/usr/bin/env python
"""
Filter benchmark problems by Diagnosis Accuracy (DA).

Research Direction: Direction A (OR-Debug-Bench) Phase 3+
Documentation: docs/plan/modules/03_BENCH.md

This script identifies "truly hard" problems where the SFT model:
- Successfully fixes the problem (high RR@5)
- But diagnoses incorrectly (low DA)

These problems are ideal for RL training because:
- There's room for improvement in understanding
- The model currently relies on shortcuts/luck
- DGRO can learn to diagnose correctly

Usage:
    # Analyze DA distribution from evaluation database
    python scripts/analysis/filter_by_da.py \
        --db outputs/experiments/2026-01-16/sft_medium_eval/results.db \
        --analyze

    # Filter problems with DA < 50%
    python scripts/analysis/filter_by_da.py \
        --benchmark data/benchmarks/or_debug_bench_medium/dataset.json \
        --db outputs/experiments/2026-01-16/sft_medium_eval/results.db \
        --da-threshold 0.5 \
        --output data/benchmarks/or_debug_bench_hard_da

Author: Ruicheng Ao
Created: 2026-01-16
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_evaluation_results(db_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation results from SQLite database.

    Args:
        db_path: Path to results.db

    Returns:
        List of result dictionaries
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT
            problem_id,
            success,
            final_status,
            steps,
            diagnosed_constraints,
            ground_truth_iis
        FROM evaluation_results
    """)

    results = []
    for row in cursor:
        result = dict(row)
        # Parse JSON fields
        if result['diagnosed_constraints']:
            try:
                result['diagnosed_constraints'] = json.loads(result['diagnosed_constraints'])
            except json.JSONDecodeError:
                result['diagnosed_constraints'] = []
        else:
            result['diagnosed_constraints'] = []

        if result['ground_truth_iis']:
            try:
                result['ground_truth_iis'] = json.loads(result['ground_truth_iis'])
            except json.JSONDecodeError:
                result['ground_truth_iis'] = []
        else:
            result['ground_truth_iis'] = []

        results.append(result)

    conn.close()
    return results


def compute_per_problem_da(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute Diagnosis Accuracy for each problem.

    DA = |diagnosed ∩ ground_truth| / |ground_truth|

    Args:
        results: List of evaluation results

    Returns:
        Dictionary mapping problem_id to DA score
    """
    da_scores = {}

    for r in results:
        problem_id = r['problem_id']
        diagnosed = set(r['diagnosed_constraints'])
        ground_truth = set(r['ground_truth_iis'])

        if not ground_truth:
            # No ground truth available
            da_scores[problem_id] = None
        else:
            intersection = diagnosed & ground_truth
            da = len(intersection) / len(ground_truth)
            da_scores[problem_id] = da

    return da_scores


def analyze_da_distribution(da_scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze the distribution of DA scores.

    Args:
        da_scores: Dictionary of problem_id -> DA

    Returns:
        Analysis report
    """
    valid_scores = [s for s in da_scores.values() if s is not None]

    if not valid_scores:
        return {"error": "No valid DA scores found"}

    import numpy as np
    scores = np.array(valid_scores)

    report = {
        "total_problems": len(da_scores),
        "valid_da_scores": len(valid_scores),
        "mean_da": float(np.mean(scores)),
        "median_da": float(np.median(scores)),
        "std_da": float(np.std(scores)),
        "min_da": float(np.min(scores)),
        "max_da": float(np.max(scores)),
        "distribution": {
            "da=0": int(np.sum(scores == 0)),
            "0<da<0.25": int(np.sum((scores > 0) & (scores < 0.25))),
            "0.25<=da<0.5": int(np.sum((scores >= 0.25) & (scores < 0.5))),
            "0.5<=da<0.75": int(np.sum((scores >= 0.5) & (scores < 0.75))),
            "0.75<=da<1.0": int(np.sum((scores >= 0.75) & (scores < 1.0))),
            "da=1.0": int(np.sum(scores == 1.0)),
        }
    }

    # Find problems in each category
    report["low_da_problems"] = [
        pid for pid, da in da_scores.items()
        if da is not None and da < 0.5
    ]
    report["high_da_problems"] = [
        pid for pid, da in da_scores.items()
        if da is not None and da >= 0.5
    ]

    return report


def filter_benchmark_by_da(
    benchmark_path: str,
    da_scores: Dict[str, float],
    da_threshold: float = 0.5,
    mode: str = "below"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter benchmark problems by DA threshold.

    Args:
        benchmark_path: Path to dataset.json
        da_scores: Dictionary of problem_id -> DA
        da_threshold: DA threshold for filtering
        mode: "below" (DA < threshold) or "above" (DA >= threshold)

    Returns:
        Tuple of (filtered_problems, excluded_problems)
    """
    with open(benchmark_path, 'r') as f:
        data = json.load(f)

    problems = data.get('problems', data)  # Handle both formats

    filtered = []
    excluded = []

    for p in problems:
        pid = p.get('problem_id', p.get('id'))
        da = da_scores.get(pid)

        if da is None:
            # No DA score, include by default
            filtered.append(p)
        elif mode == "below" and da < da_threshold:
            filtered.append(p)
        elif mode == "above" and da >= da_threshold:
            filtered.append(p)
        else:
            excluded.append(p)

    return filtered, excluded


def create_filtered_benchmark(
    source_benchmark: str,
    filtered_problems: List[Dict],
    output_dir: str,
    metadata: Dict[str, Any]
):
    """
    Create a new benchmark directory with filtered problems.

    Args:
        source_benchmark: Path to source benchmark directory
        filtered_problems: List of filtered problem dictionaries
        output_dir: Path to output directory
        metadata: Additional metadata to include
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load source benchmark
    source_path = Path(source_benchmark)
    if source_path.is_dir():
        dataset_file = source_path / "dataset.json"
    else:
        dataset_file = source_path
        source_path = source_path.parent

    with open(dataset_file, 'r') as f:
        original_data = json.load(f)

    # Create filtered dataset
    filtered_data = {
        "benchmark_name": f"or_debug_bench_hard_da",
        "description": f"Problems filtered by DA < {metadata.get('da_threshold', 0.5)}",
        "filter_criteria": metadata,
        "n_problems": len(filtered_problems),
        "source_benchmark": str(source_benchmark),
        "problems": filtered_problems
    }

    # Save dataset
    with open(output_path / "dataset.json", 'w') as f:
        json.dump(filtered_data, f, indent=2)

    # Copy MPS files if they exist and are referenced
    mps_dir = source_path / "mps_files"
    if mps_dir.exists():
        output_mps_dir = output_path / "mps_files"
        output_mps_dir.mkdir(exist_ok=True)

        for p in filtered_problems:
            mps_file = p.get('mps_file', '')
            if mps_file:
                src_mps = source_path / mps_file
                if src_mps.exists():
                    dst_mps = output_path / mps_file
                    dst_mps.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_mps, dst_mps)

    print(f"Created filtered benchmark at {output_path}")
    print(f"  - {len(filtered_problems)} problems")
    print(f"  - Filter: DA < {metadata.get('da_threshold', 0.5)}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter benchmark problems by Diagnosis Accuracy"
    )
    parser.add_argument(
        "--db", type=str, required=True,
        help="Path to evaluation results database"
    )
    parser.add_argument(
        "--benchmark", type=str,
        help="Path to benchmark dataset.json"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output directory for filtered benchmark"
    )
    parser.add_argument(
        "--da-threshold", type=float, default=0.5,
        help="DA threshold for filtering (default: 0.5)"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Only analyze DA distribution, don't filter"
    )
    parser.add_argument(
        "--mode", choices=["below", "above"], default="below",
        help="Filter mode: below (DA < threshold) or above (default: below)"
    )

    args = parser.parse_args()

    # Load evaluation results
    print(f"Loading results from {args.db}...")
    results = load_evaluation_results(args.db)
    print(f"  Loaded {len(results)} evaluation results")

    # Compute per-problem DA
    da_scores = compute_per_problem_da(results)

    # Analyze distribution
    analysis = analyze_da_distribution(da_scores)

    print("\n=== DA Distribution Analysis ===")
    print(f"Total problems: {analysis['total_problems']}")
    print(f"Valid DA scores: {analysis['valid_da_scores']}")
    print(f"Mean DA: {analysis['mean_da']:.2%}")
    print(f"Median DA: {analysis['median_da']:.2%}")
    print(f"Std DA: {analysis['std_da']:.2%}")
    print(f"Min/Max DA: {analysis['min_da']:.2%} / {analysis['max_da']:.2%}")
    print("\nDistribution:")
    for bucket, count in analysis['distribution'].items():
        pct = count / analysis['valid_da_scores'] * 100 if analysis['valid_da_scores'] > 0 else 0
        print(f"  {bucket}: {count} ({pct:.1f}%)")

    print(f"\nLow DA problems (DA < 0.5): {len(analysis['low_da_problems'])}")
    print(f"High DA problems (DA >= 0.5): {len(analysis['high_da_problems'])}")

    if args.analyze:
        # Save analysis to JSON
        output_file = Path(args.db).parent / "da_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to {output_file}")
        return

    # Filter benchmark
    if not args.benchmark:
        print("\nError: --benchmark required for filtering")
        return

    if not args.output:
        print("\nError: --output required for filtering")
        return

    print(f"\nFiltering benchmark with DA < {args.da_threshold}...")
    filtered, excluded = filter_benchmark_by_da(
        args.benchmark,
        da_scores,
        args.da_threshold,
        args.mode
    )

    print(f"  Filtered: {len(filtered)} problems")
    print(f"  Excluded: {len(excluded)} problems")

    # Create filtered benchmark
    metadata = {
        "da_threshold": args.da_threshold,
        "mode": args.mode,
        "source_db": args.db,
        "filtered_count": len(filtered),
        "excluded_count": len(excluded),
        "mean_da_before": analysis['mean_da'],
    }

    create_filtered_benchmark(
        args.benchmark,
        filtered,
        args.output,
        metadata
    )


if __name__ == "__main__":
    main()
