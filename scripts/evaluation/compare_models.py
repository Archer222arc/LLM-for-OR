#!/home/Archer/miniforge3/bin/python
"""
Compare SFT and GRPO model evaluation results.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/04_EVAL.md

Usage:
    python scripts/evaluation/compare_models.py \
        --sft outputs/sft_eval_200.json \
        --grpo outputs/grpo_eval_200.json \
        --output outputs/sft_vs_grpo_comparison.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_results(path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_metrics_from_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute metrics from raw results."""
    n = len(results)
    if n == 0:
        return {}

    successes = sum(1 for r in results if r['success'])
    recovery_rate = successes / n

    # RR@k calculations
    rr_at = {}
    for k in [5, 10, 15, 20]:
        count = sum(1 for r in results if r.get('success_at_step') and r['success_at_step'] <= k)
        rr_at[f'rr_at_{k}'] = count / n

    # Average steps for successful episodes
    successful_steps = [r['steps'] for r in results if r['success']]
    avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0

    # Token statistics
    total_tokens = sum(r.get('total_tokens', 0) for r in results)
    avg_tokens = total_tokens / n if n > 0 else 0

    return {
        'n_episodes': n,
        'recovery_rate': recovery_rate,
        'avg_steps': avg_steps,
        **rr_at,
        'avg_tokens': avg_tokens,
        'total_tokens': total_tokens,
    }


def generate_comparison_report(
    sft_data: Dict[str, Any],
    grpo_data: Dict[str, Any],
) -> str:
    """Generate markdown comparison report."""

    sft_summary = sft_data.get('summary', compute_metrics_from_results(sft_data.get('results', [])))
    grpo_summary = grpo_data.get('summary', compute_metrics_from_results(grpo_data.get('results', [])))

    sft_name = sft_data.get('model_name', 'SFT')
    grpo_name = grpo_data.get('model_name', 'GRPO')

    lines = [
        "# SFT vs GRPO Model Comparison",
        "",
        "## Overview",
        "",
        f"- **SFT Model**: {sft_data.get('model', 'unknown')}",
        f"- **GRPO Model**: {grpo_data.get('model', 'unknown')}",
        f"- **Dataset**: {sft_data.get('dataset', 'unknown')}",
        f"- **Samples**: {sft_data.get('num_problems', 0)}",
        "",
        "## Main Results",
        "",
        "| Metric | SFT | GRPO | Delta |",
        "|--------|-----|------|-------|",
    ]

    # Main metrics comparison
    metrics = [
        ('Recovery Rate', 'recovery_rate', True),
        ('RR@5', 'rr_at_5', True),
        ('RR@10', 'rr_at_10', True),
        ('RR@15', 'rr_at_15', True),
        ('RR@20', 'rr_at_20', True),
        ('Avg Steps', 'avg_steps', False),
        ('Diagnosis Accuracy', 'diagnosis_accuracy', True),
        ('Token Efficiency', 'token_efficiency', True),
    ]

    for label, key, is_rate in metrics:
        sft_val = sft_summary.get(key, 0)
        grpo_val = grpo_summary.get(key, 0)

        if sft_val is None:
            sft_val = 0
        if grpo_val is None:
            grpo_val = 0

        delta = grpo_val - sft_val

        if is_rate:
            sft_str = f"{sft_val*100:.1f}%"
            grpo_str = f"{grpo_val*100:.1f}%"
            delta_str = f"{delta*100:+.1f}%"
        else:
            sft_str = f"{sft_val:.2f}"
            grpo_str = f"{grpo_val:.2f}"
            delta_str = f"{delta:+.2f}"

        lines.append(f"| {label} | {sft_str} | {grpo_str} | {delta_str} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    # Analysis
    rr_sft = sft_summary.get('recovery_rate', 0)
    rr_grpo = grpo_summary.get('recovery_rate', 0)
    rr_delta = rr_grpo - rr_sft

    if abs(rr_delta) < 0.02:
        lines.append("**Finding**: SFT and GRPO models show similar performance.")
        lines.append("")
        lines.append("This result is expected given that GRPO training showed zero reward variance,")
        lines.append("indicating the SFT model was already well-optimized for this task.")
        lines.append("")
        lines.append("**Implication**: For this domain, SFT alone achieves strong performance.")
        lines.append("GRPO may benefit from:")
        lines.append("- Higher temperature during training for diversity")
        lines.append("- Larger, more diverse training dataset")
        lines.append("- Different reward signal design")
    elif rr_delta > 0.05:
        lines.append(f"**Finding**: GRPO outperforms SFT by {rr_delta*100:.1f}% on recovery rate.")
        lines.append("")
        lines.append("The RL training successfully improved the model's debugging capability.")
    else:
        lines.append(f"**Finding**: GRPO underperforms SFT by {abs(rr_delta)*100:.1f}% on recovery rate.")
        lines.append("")
        lines.append("This may indicate:")
        lines.append("- Reward function misalignment")
        lines.append("- Training instability")
        lines.append("- Need for hyperparameter tuning")

    lines.extend([
        "",
        "## Per-Problem Comparison",
        "",
        "| Problem Type | SFT RR | GRPO RR |",
        "|--------------|--------|---------|",
    ])

    # Per-type breakdown if available
    for error_type in ['A', 'B', 'C', 'D']:
        sft_results = sft_data.get('results', [])
        grpo_results = grpo_data.get('results', [])

        sft_type = [r for r in sft_results if error_type in r.get('problem_id', '')]
        grpo_type = [r for r in grpo_results if error_type in r.get('problem_id', '')]

        if sft_type and grpo_type:
            sft_rr = sum(1 for r in sft_type if r['success']) / len(sft_type) if sft_type else 0
            grpo_rr = sum(1 for r in grpo_type if r['success']) / len(grpo_type) if grpo_type else 0
            lines.append(f"| Type {error_type} | {sft_rr*100:.1f}% | {grpo_rr*100:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        f"Generated by `scripts/evaluation/compare_models.py`",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare SFT and GRPO model results")
    parser.add_argument("--sft", type=str, required=True, help="Path to SFT evaluation results")
    parser.add_argument("--grpo", type=str, required=True, help="Path to GRPO evaluation results")
    parser.add_argument("--output", type=str, required=True, help="Output path for comparison report")
    args = parser.parse_args()

    print(f"Loading SFT results from: {args.sft}")
    sft_data = load_results(args.sft)

    print(f"Loading GRPO results from: {args.grpo}")
    grpo_data = load_results(args.grpo)

    print("Generating comparison report...")
    report = generate_comparison_report(sft_data, grpo_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {args.output}")
    print()
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
