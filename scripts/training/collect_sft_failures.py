#!/usr/bin/env python3
"""
Collect SFT Failure Cases for Curriculum Learning.

This script extracts failure cases from SFT evaluation results to create
a focused training set for GRPO curriculum learning.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-15_phase2_grpo_improvements.md
Phase: 2.1 - Curriculum Learning on Failures

Key Components:
    - FailureCase: Dataclass for structured failure representation
    - load_failures_from_db: Extract failures from SQLite database
    - analyze_failure_patterns: Statistical analysis of failure modes
    - generate_weakness_profile: Create targeting profile for Saboteur

Example:
    >>> python scripts/training/collect_sft_failures.py \\
    ...     --db outputs/experiments/2026-01-15/sft_holdout_eval/results.db \\
    ...     --model sft \\
    ...     --output data/training/sft_failures.json

Output:
    JSON file containing:
    - List of failure problem IDs
    - Failure pattern analysis (type distribution, IIS sizes)
    - Weakness profile for Saboteur targeting
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class FailureCase:
    """Represents a single failure case."""
    problem_id: str
    final_status: str
    steps: int
    diagnosed_constraints: List[str]
    ground_truth_iis: List[str]
    total_reward: float
    failure_type: str  # 'timeout', 'wrong_fix', 'wrong_diagnosis', 'other'
    error_type: Optional[str] = None  # A, B, C, D, E, F, G, H, I
    iis_size: int = 0


def extract_error_type(problem_id: str) -> Optional[str]:
    """Extract error type from problem ID (e.g., 'A_001' -> 'A')."""
    if '_' in problem_id:
        parts = problem_id.split('_')
        if len(parts) >= 1 and len(parts[0]) == 1 and parts[0].isalpha():
            return parts[0].upper()
    return None


def classify_failure(
    success: bool,
    steps: int,
    max_steps: int,
    final_status: str,
    diagnosed: List[str],
    ground_truth: List[str]
) -> str:
    """
    Classify the type of failure.

    Returns:
        'timeout': Ran out of steps
        'wrong_fix': Fixed but wrong status
        'wrong_diagnosis': All diagnoses wrong
        'partial_diagnosis': Some diagnoses wrong
        'other': Other failure types
    """
    if not success:
        if steps >= max_steps:
            return 'timeout'
        if final_status == 'INFEASIBLE':
            # Check if diagnosis was correct
            if diagnosed and ground_truth:
                overlap = set(diagnosed) & set(ground_truth)
                if len(overlap) == 0:
                    return 'wrong_diagnosis'
                elif len(overlap) < len(diagnosed):
                    return 'partial_diagnosis'
            return 'wrong_fix'
        return 'other'
    return 'success'


def load_failures_from_db(
    db_path: str,
    model_name: str,
    max_steps: int = 20,
    include_slow_success: bool = True,
    slow_threshold: int = 5
) -> List[FailureCase]:
    """
    Load failure cases from SQLite database.

    Args:
        db_path: Path to SQLite database
        model_name: Model name in the database
        max_steps: Maximum steps considered (failures at this point are timeouts)
        include_slow_success: Include problems that succeeded but took > slow_threshold steps
        slow_threshold: Steps threshold for "slow" success

    Returns:
        List of FailureCase objects
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query all results for the model
    cursor.execute("""
        SELECT problem_id, success, final_status, steps, total_reward,
               diagnosed_constraints, ground_truth_iis
        FROM evaluation_results
        WHERE model_name = ?
    """, (model_name,))

    failures = []
    for row in cursor.fetchall():
        problem_id, success, final_status, steps, reward, diag_json, iis_json = row

        diagnosed = json.loads(diag_json) if diag_json else []
        ground_truth = json.loads(iis_json) if iis_json else []

        # Determine if this is a failure case
        is_failure = not success
        is_slow_success = success and steps > slow_threshold

        if is_failure or (include_slow_success and is_slow_success):
            failure_type = classify_failure(
                success, steps, max_steps, final_status, diagnosed, ground_truth
            )

            if failure_type != 'success':
                failures.append(FailureCase(
                    problem_id=problem_id,
                    final_status=final_status,
                    steps=steps,
                    diagnosed_constraints=diagnosed,
                    ground_truth_iis=ground_truth,
                    total_reward=reward,
                    failure_type=failure_type,
                    error_type=extract_error_type(problem_id),
                    iis_size=len(ground_truth)
                ))

    conn.close()
    return failures


def analyze_failure_patterns(failures: List[FailureCase]) -> Dict[str, Any]:
    """
    Analyze patterns in failure cases.

    Returns:
        Dictionary with failure pattern statistics
    """
    if not failures:
        return {"message": "No failures found"}

    # Count by failure type
    type_counts = Counter(f.failure_type for f in failures)

    # Count by error type
    error_type_counts = Counter(f.error_type for f in failures if f.error_type)

    # IIS size distribution
    iis_sizes = [f.iis_size for f in failures]
    avg_iis_size = sum(iis_sizes) / len(iis_sizes) if iis_sizes else 0

    # Steps distribution for failures
    steps_dist = [f.steps for f in failures]
    avg_steps = sum(steps_dist) / len(steps_dist) if steps_dist else 0

    # Identify problematic error types
    error_type_failure_rate = {}
    for error_type in error_type_counts:
        error_type_failure_rate[error_type] = {
            'count': error_type_counts[error_type],
            'proportion': error_type_counts[error_type] / len(failures)
        }

    return {
        'total_failures': len(failures),
        'failure_type_distribution': dict(type_counts),
        'error_type_distribution': dict(error_type_counts),
        'error_type_failure_rate': error_type_failure_rate,
        'avg_iis_size': avg_iis_size,
        'max_iis_size': max(iis_sizes) if iis_sizes else 0,
        'min_iis_size': min(iis_sizes) if iis_sizes else 0,
        'avg_steps_at_failure': avg_steps,
        'iis_size_percentiles': {
            'p25': sorted(iis_sizes)[len(iis_sizes)//4] if iis_sizes else 0,
            'p50': sorted(iis_sizes)[len(iis_sizes)//2] if iis_sizes else 0,
            'p75': sorted(iis_sizes)[3*len(iis_sizes)//4] if iis_sizes else 0,
        }
    }


def generate_weakness_profile(failures: List[FailureCase]) -> Dict[str, Any]:
    """
    Generate a weakness profile for Saboteur to target.

    This profile can be used to generate more problems that target
    the model's weaknesses.
    """
    analysis = analyze_failure_patterns(failures)

    # Find the most common failure patterns
    weakness_profile = {
        'target_iis_size': {
            'min': analysis.get('iis_size_percentiles', {}).get('p50', 3),
            'prefer_large': True
        },
        'target_error_types': [],
        'avoid_error_types': [],
        'constraint_patterns': []
    }

    # Identify which error types cause most failures
    error_rates = analysis.get('error_type_failure_rate', {})
    for error_type, stats in sorted(error_rates.items(),
                                     key=lambda x: x[1]['count'],
                                     reverse=True):
        weakness_profile['target_error_types'].append(error_type)

    # Analyze constraint patterns in failures
    constraint_patterns = defaultdict(int)
    for f in failures:
        for constr in f.ground_truth_iis:
            # Extract pattern from constraint name
            if '_' in constr:
                pattern = constr.rsplit('_', 1)[0]
                constraint_patterns[pattern] += 1

    # Top constraint patterns
    top_patterns = sorted(constraint_patterns.items(),
                          key=lambda x: x[1], reverse=True)[:10]
    weakness_profile['constraint_patterns'] = [p[0] for p in top_patterns]

    return weakness_profile


def main():
    parser = argparse.ArgumentParser(
        description="Collect SFT failure cases for curriculum learning"
    )
    parser.add_argument(
        '--db', type=str, required=True,
        help='Path to SQLite database with evaluation results'
    )
    parser.add_argument(
        '--model', type=str, default='sft',
        help='Model name in the database (default: sft)'
    )
    parser.add_argument(
        '--output', type=str, default='data/training/sft_failures.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--max-steps', type=int, default=20,
        help='Maximum steps considered (default: 20)'
    )
    parser.add_argument(
        '--slow-threshold', type=int, default=5,
        help='Steps threshold for slow success (default: 5)'
    )
    parser.add_argument(
        '--include-slow', action='store_true', default=True,
        help='Include slow successes (>threshold steps) as failures'
    )
    parser.add_argument(
        '--no-include-slow', action='store_false', dest='include_slow',
        help='Do not include slow successes'
    )

    args = parser.parse_args()

    # Check database exists
    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)

    print(f"Loading failures from: {args.db}")
    print(f"Model: {args.model}")

    # Load failures
    failures = load_failures_from_db(
        db_path=args.db,
        model_name=args.model,
        max_steps=args.max_steps,
        include_slow_success=args.include_slow,
        slow_threshold=args.slow_threshold
    )

    print(f"Found {len(failures)} failure cases")

    if not failures:
        print("No failures found. Check model name and database.")
        sys.exit(0)

    # Analyze patterns
    patterns = analyze_failure_patterns(failures)
    weakness_profile = generate_weakness_profile(failures)

    # Prepare output
    output = {
        'metadata': {
            'source_db': str(args.db),
            'model': args.model,
            'max_steps': args.max_steps,
            'slow_threshold': args.slow_threshold,
            'include_slow_success': args.include_slow,
            'timestamp': datetime.now().isoformat(),
        },
        'summary': {
            'total_failures': len(failures),
            'patterns': patterns,
        },
        'weakness_profile': weakness_profile,
        'failures': [asdict(f) for f in failures],
        'problem_ids': [f.problem_id for f in failures],
    }

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("FAILURE ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nTotal failures: {len(failures)}")
    print(f"\nFailure type distribution:")
    for ftype, count in patterns.get('failure_type_distribution', {}).items():
        print(f"  {ftype}: {count} ({count/len(failures)*100:.1f}%)")

    print(f"\nError type distribution:")
    for etype, count in sorted(patterns.get('error_type_distribution', {}).items()):
        print(f"  Type {etype}: {count}")

    print(f"\nIIS size statistics:")
    print(f"  Average: {patterns.get('avg_iis_size', 0):.1f}")
    print(f"  Min: {patterns.get('min_iis_size', 0)}")
    print(f"  Max: {patterns.get('max_iis_size', 0)}")

    print(f"\nWeakness profile (for Saboteur targeting):")
    print(f"  Target error types: {weakness_profile['target_error_types']}")
    print(f"  Target IIS size >= {weakness_profile['target_iis_size']['min']}")

    print("\n" + "="*60)
    print("Next step: Use this data for curriculum GRPO training")
    print("="*60)


if __name__ == '__main__':
    main()
