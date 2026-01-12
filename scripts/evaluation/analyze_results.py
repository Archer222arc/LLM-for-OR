#!/usr/bin/env python3
"""
Results Analysis Script for OR-Debug-Bench.

Analyzes evaluation results and generates tables and visualizations
for the paper.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/04_EVAL.md

Usage:
    python scripts/evaluation/analyze_results.py --input outputs/baseline_results.json

    # Generate all tables
    python scripts/evaluation/analyze_results.py --input outputs/full_eval_results.json --output outputs/analysis/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path (scripts/evaluation -> project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_main_results_table(results: Dict[str, Any]) -> str:
    """
    Generate main results table (Table 1 in paper).

    Args:
        results: Results dictionary from evaluate_llm.py

    Returns:
        Formatted markdown table
    """
    lines = [
        "## Table 1: Main Results",
        "",
        "| Model | Type | RR | RR@5 | RR@10 | RR@20 | DA | OP | Steps |",
        "|-------|------|-----|------|-------|-------|-----|-----|-------|",
    ]

    agent_results = results.get('results', [])

    for r in agent_results:
        name = r.get('agent_name', 'unknown')
        agent_type = _get_agent_type(name)
        rr = r.get('recovery_rate', 0)
        rr5 = r.get('rr_at_5', 0)
        rr10 = r.get('rr_at_10', 0)
        rr20 = r.get('rr_at_20', 0)
        da = r.get('diagnosis_accuracy', 0) or 0
        op = r.get('optimality_preservation', 0) or 0
        steps = r.get('avg_steps', 0)

        lines.append(
            f"| {name} | {agent_type} | {rr:.1%} | {rr5:.1%} | {rr10:.1%} | {rr20:.1%} | {da:.1%} | {op:.1%} | {steps:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)


def print_error_type_table(results: Dict[str, Any]) -> str:
    """
    Generate error type breakdown table (Table 2 in paper).

    Args:
        results: Results dictionary

    Returns:
        Formatted markdown table
    """
    lines = [
        "## Table 2: Results by Error Type",
        "",
        "| Model | Type A | Type B | Type C | Type D |",
        "|-------|--------|--------|--------|--------|",
    ]

    agent_results = results.get('results', [])

    for r in agent_results:
        name = r.get('agent_name', 'unknown')
        by_type = r.get('by_error_type', {})

        type_a = by_type.get('A', {}).get('recovery_rate', 0)
        type_b = by_type.get('B', {}).get('recovery_rate', 0)
        type_c = by_type.get('C', {}).get('recovery_rate', 0)
        type_d = by_type.get('D', {}).get('recovery_rate', 0)

        lines.append(f"| {name} | {type_a:.1%} | {type_b:.1%} | {type_c:.1%} | {type_d:.1%} |")

    lines.append("")
    return "\n".join(lines)


def print_difficulty_table(results: Dict[str, Any]) -> str:
    """
    Generate difficulty breakdown table.

    Args:
        results: Results dictionary

    Returns:
        Formatted markdown table
    """
    lines = [
        "## Table 3: Results by Difficulty",
        "",
        "| Model | Easy | Medium | Hard |",
        "|-------|------|--------|------|",
    ]

    agent_results = results.get('results', [])

    for r in agent_results:
        name = r.get('agent_name', 'unknown')
        by_diff = r.get('by_difficulty', {})

        easy = by_diff.get('easy', {}).get('recovery_rate', 0)
        medium = by_diff.get('medium', {}).get('recovery_rate', 0)
        hard = by_diff.get('hard', {}).get('recovery_rate', 0)

        lines.append(f"| {name} | {easy:.1%} | {medium:.1%} | {hard:.1%} |")

    lines.append("")
    return "\n".join(lines)


def print_rr_at_k_table(results: Dict[str, Any]) -> str:
    """
    Generate RR@k table for test-time compute analysis.

    Args:
        results: Results dictionary

    Returns:
        Formatted markdown table
    """
    lines = [
        "## Table 4: Test-Time Compute (RR@k)",
        "",
        "| Model | RR@5 | RR@10 | RR@15 | RR@20 |",
        "|-------|------|-------|-------|-------|",
    ]

    agent_results = results.get('results', [])

    for r in agent_results:
        name = r.get('agent_name', 'unknown')
        rr5 = r.get('rr_at_5', 0)
        rr10 = r.get('rr_at_10', 0)
        rr15 = r.get('rr_at_15', 0)
        rr20 = r.get('rr_at_20', 0)

        lines.append(f"| {name} | {rr5:.1%} | {rr10:.1%} | {rr15:.1%} | {rr20:.1%} |")

    lines.append("")
    return "\n".join(lines)


def print_summary_stats(results: Dict[str, Any]) -> str:
    """
    Print summary statistics.

    Args:
        results: Results dictionary

    Returns:
        Formatted summary string
    """
    lines = [
        "## Summary Statistics",
        "",
        f"- **Dataset**: {results.get('dataset', 'unknown')}",
        f"- **Problems**: {results.get('n_problems', 0)}",
        f"- **Max Steps**: {results.get('max_steps', 0)}",
        f"- **Timestamp**: {results.get('timestamp', 'unknown')}",
        "",
    ]

    # Find best model
    agent_results = results.get('results', [])
    if agent_results:
        best = max(agent_results, key=lambda x: x.get('recovery_rate', 0))
        lines.append(f"- **Best Model**: {best.get('agent_name', 'unknown')} (RR={best.get('recovery_rate', 0):.1%})")

    lines.append("")
    return "\n".join(lines)


def generate_test_time_compute_data(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate data for test-time compute plot.

    Args:
        results: Results dictionary

    Returns:
        List of data points for plotting
    """
    data = []
    agent_results = results.get('results', [])

    for r in agent_results:
        name = r.get('agent_name', 'unknown')
        for k in [5, 10, 15, 20]:
            rr_k = r.get(f'rr_at_{k}', 0)
            data.append({
                'model': name,
                'k': k,
                'rr': rr_k,
            })

    return data


def _get_agent_type(agent_name: str) -> str:
    """Determine agent type category."""
    name_lower = agent_name.lower()

    if any(x in name_lower for x in ['heuristic', 'greedy', 'random', 'donothing']):
        return 'Baseline'
    elif any(x in name_lower for x in ['o1', 'o4', 'deepseek-r1', 'thinking']):
        return 'Reasoning'
    elif any(x in name_lower for x in ['gpt', 'claude', 'qwen', 'deepseek']):
        return 'General'
    else:
        return 'Other'


def main():
    parser = argparse.ArgumentParser(
        description="Analyze OR-Debug-Bench evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for analysis files'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='markdown',
        choices=['markdown', 'latex', 'csv'],
        help='Output format'
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.input)

    # Generate all tables
    output_lines = [
        "# OR-Debug-Bench Evaluation Analysis",
        "",
    ]

    output_lines.append(print_summary_stats(results))
    output_lines.append(print_main_results_table(results))
    output_lines.append(print_error_type_table(results))
    output_lines.append(print_difficulty_table(results))
    output_lines.append(print_rr_at_k_table(results))

    # Print to stdout
    output_text = "\n".join(output_lines)
    print(output_text)

    # Save to file if output path provided
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown
        md_path = output_dir / "analysis.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\nSaved analysis to: {md_path}")

        # Save test-time compute data
        ttc_data = generate_test_time_compute_data(results)
        ttc_path = output_dir / "test_time_compute.json"
        with open(ttc_path, 'w', encoding='utf-8') as f:
            json.dump(ttc_data, f, indent=2)
        print(f"Saved TTC data to: {ttc_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
