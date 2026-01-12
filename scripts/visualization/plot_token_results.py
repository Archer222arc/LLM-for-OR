#!/usr/bin/env python3
"""
Visualization script for OR-Debug-Bench Token Tracking Results.

Generates publication-quality figures for model comparison.
All labels and text in English.

Usage:
    python scripts/visualization/plot_token_results.py \
        --db-dir outputs/experiments/2026-01-12_token_tracking \
        --output outputs/experiments/2026-01-12_token_tracking/figures
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
FIGSIZE = (12, 7)
DPI = 300
FONT_SIZE = 12

# Model display order (by efficiency)
MODEL_ORDER = [
    'gpt-5.2-chat',
    'o4-mini',
    'o1',
    'Kimi-K2-Thinking',
    'gpt-5-mini',
    'Llama-3.3-70B-Instruct',
    'gpt-4.1-mini',
    'gpt-5-nano'
]

# Color mapping for models
MODEL_COLORS = {
    'gpt-5.2-chat': '#2ecc71',       # Green - best
    'o4-mini': '#3498db',             # Blue
    'o1': '#9b59b6',                  # Purple
    'Kimi-K2-Thinking': '#1abc9c',    # Teal
    'gpt-5-mini': '#f39c12',          # Orange
    'Llama-3.3-70B-Instruct': '#e74c3c',  # Red
    'gpt-4.1-mini': '#95a5a6',        # Gray
    'gpt-5-nano': '#7f8c8d'           # Dark gray - worst
}


def load_all_results(db_dir: str) -> pd.DataFrame:
    """Load results from all SQLite databases in directory."""
    db_path = Path(db_dir)
    all_data = []

    for db_file in db_path.glob("*.db"):
        model_name = db_file.stem
        conn = sqlite3.connect(db_file)

        try:
            df = pd.read_sql_query(
                "SELECT * FROM evaluation_results",
                conn
            )
            df['model'] = model_name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {db_file}: {e}")
        finally:
            conn.close()

    if not all_data:
        raise ValueError(f"No data found in {db_dir}")

    return pd.concat(all_data, ignore_index=True)


def compute_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics per model."""
    stats = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        n = len(model_df)
        successes = model_df['success'].sum()

        stat = {
            'model': model,
            'n': n,
            'rr': 100.0 * successes / n,
            'avg_steps': model_df['steps'].mean(),
            'avg_tokens': model_df['total_tokens'].mean(),
            'avg_input_tokens': model_df['input_tokens'].mean(),
            'avg_output_tokens': model_df['output_tokens'].mean(),
            'avg_reasoning_tokens': model_df['reasoning_tokens'].mean(),
            'avg_api_calls': model_df['api_call_count'].mean(),
            'avg_wall_clock': model_df['wall_clock_seconds'].mean(),
            'tokens_per_step': model_df['total_tokens'].mean() / model_df['steps'].mean(),
        }

        # Token efficiency
        stat['efficiency'] = (stat['rr'] / 100) * 1000 / stat['avg_tokens'] if stat['avg_tokens'] > 0 else 0

        # RR@k steps
        for k in [3, 5, 10, 15, 20]:
            rr_at_k = 100.0 * model_df[(model_df['success'] == 1) & (model_df['steps'] <= k)].shape[0] / n
            stat[f'rr_at_{k}_steps'] = rr_at_k

        # RR@token budget
        for budget in [5000, 10000, 15000, 20000, 50000]:
            rr_at_budget = 100.0 * model_df[(model_df['success'] == 1) & (model_df['total_tokens'] <= budget)].shape[0] / n
            stat[f'rr_at_{budget}_tokens'] = rr_at_budget

        stats.append(stat)

    stats_df = pd.DataFrame(stats)

    # Sort by efficiency
    stats_df['model'] = pd.Categorical(stats_df['model'], categories=MODEL_ORDER, ordered=True)
    stats_df = stats_df.sort_values('model').reset_index(drop=True)

    return stats_df


def plot_performance_overview(stats: pd.DataFrame, output: str):
    """Figure 1: Model Performance Overview (Bar Chart)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    models = stats['model'].tolist()
    rr_values = stats['rr'].tolist()
    tokens = stats['avg_tokens'].tolist()
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in models]

    bars = ax.bar(range(len(models)), rr_values, color=colors, edgecolor='black', linewidth=0.5)

    # Add token annotations on bars
    for i, (bar, tok) in enumerate(zip(bars, tokens)):
        height = bar.get_height()
        ax.annotate(f'{tok/1000:.1f}k tokens',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, rotation=0)

    ax.set_xlabel('Model', fontsize=FONT_SIZE)
    ax.set_ylabel('Recovery Rate (%)', fontsize=FONT_SIZE)
    ax.set_title('OR-Debug-Bench: Model Performance Overview', fontsize=FONT_SIZE + 2, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')

    plt.tight_layout()
    plt.savefig(output, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_token_efficiency(stats: pd.DataFrame, output: str):
    """Figure 2: Token Efficiency Scatter Plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for _, row in stats.iterrows():
        model = row['model']
        color = MODEL_COLORS.get(model, '#95a5a6')
        size = row['avg_steps'] * 30  # Scale point size by steps

        ax.scatter(row['avg_tokens'], row['rr'],
                   s=size, c=color, alpha=0.7, edgecolors='black', linewidth=1)

        # Label offset to avoid overlap
        offset_x = 0.05 if row['avg_tokens'] < 20000 else -0.05
        ha = 'left' if row['avg_tokens'] < 20000 else 'right'

        ax.annotate(model, (row['avg_tokens'], row['rr']),
                    xytext=(offset_x, 0.02), textcoords='axes fraction',
                    fontsize=9, alpha=0.9,
                    arrowprops=dict(arrowstyle='-', alpha=0.3))

    # Re-annotate without arrows for clarity
    for _, row in stats.iterrows():
        model = row['model']
        ax.annotate(model, (row['avg_tokens'], row['rr']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.9)

    ax.set_xscale('log')
    ax.set_xlabel('Average Tokens (log scale)', fontsize=FONT_SIZE)
    ax.set_ylabel('Recovery Rate (%)', fontsize=FONT_SIZE)
    ax.set_title('Token Efficiency: Recovery Rate vs Token Cost', fontsize=FONT_SIZE + 2, fontweight='bold')

    # Highlight efficient region
    ax.axvspan(0, 10000, alpha=0.1, color='green', label='Efficient region (<10k tokens)')
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left')

    ax.set_xlim(4000, 100000)
    ax.set_ylim(40, 105)

    plt.tight_layout()
    plt.savefig(output, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_rr_at_steps(stats: pd.DataFrame, output: str):
    """Figure 3: RR@Steps Line Chart (Test-Time Compute)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    steps = [3, 5, 10, 15, 20]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    for i, (_, row) in enumerate(stats.iterrows()):
        model = row['model']
        color = MODEL_COLORS.get(model, '#95a5a6')
        rr_values = [row[f'rr_at_{k}_steps'] for k in steps]

        ax.plot(steps, rr_values, marker=markers[i % len(markers)],
                color=color, linewidth=2, markersize=8, label=model)

    ax.set_xlabel('Step Budget (k)', fontsize=FONT_SIZE)
    ax.set_ylabel('Recovery Rate (%)', fontsize=FONT_SIZE)
    ax.set_title('Test-Time Compute: RR@Steps', fontsize=FONT_SIZE + 2, fontweight='bold')
    ax.set_xticks(steps)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_rr_at_tokens(stats: pd.DataFrame, output: str):
    """Figure 4: RR@TokenBudget Line Chart."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    budgets = [5000, 10000, 15000, 20000, 50000]
    budget_labels = ['5k', '10k', '15k', '20k', '50k']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    for i, (_, row) in enumerate(stats.iterrows()):
        model = row['model']
        color = MODEL_COLORS.get(model, '#95a5a6')
        rr_values = [row[f'rr_at_{b}_tokens'] for b in budgets]

        ax.plot(range(len(budgets)), rr_values, marker=markers[i % len(markers)],
                color=color, linewidth=2, markersize=8, label=model)

    ax.set_xlabel('Token Budget', fontsize=FONT_SIZE)
    ax.set_ylabel('Recovery Rate (%)', fontsize=FONT_SIZE)
    ax.set_title('Test-Time Compute: RR@TokenBudget', fontsize=FONT_SIZE + 2, fontweight='bold')
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(budget_labels)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_token_composition(stats: pd.DataFrame, output: str):
    """Figure 5: Token Composition Stacked Bar."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    models = stats['model'].tolist()
    x = np.arange(len(models))
    width = 0.6

    # Calculate components
    input_tokens = stats['avg_input_tokens'].values
    reasoning_tokens = stats['avg_reasoning_tokens'].values
    output_only = stats['avg_output_tokens'].values - reasoning_tokens  # Non-reasoning output
    output_only = np.maximum(output_only, 0)  # Ensure non-negative

    # Stacked bars
    p1 = ax.bar(x, input_tokens, width, label='Input Tokens', color='#3498db')
    p2 = ax.bar(x, output_only, width, bottom=input_tokens, label='Output Tokens', color='#2ecc71')
    p3 = ax.bar(x, reasoning_tokens, width, bottom=input_tokens + output_only,
                label='Reasoning Tokens', color='#e74c3c')

    ax.set_xlabel('Model', fontsize=FONT_SIZE)
    ax.set_ylabel('Token Count', fontsize=FONT_SIZE)
    ax.set_title('Token Composition by Model', fontsize=FONT_SIZE + 2, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')

    # Add total annotations
    totals = input_tokens + output_only + reasoning_tokens
    for i, total in enumerate(totals):
        ax.annotate(f'{total/1000:.1f}k',
                    xy=(i, total),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_efficiency_heatmap(stats: pd.DataFrame, output: str):
    """Figure 6: Efficiency Heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Select metrics for heatmap
    metrics = ['rr', 'rr_at_5_steps', 'rr_at_10000_tokens', 'efficiency', 'tokens_per_step']
    metric_labels = ['Recovery Rate\n(%)', 'RR@5 Steps\n(%)', 'RR@10k Tokens\n(%)',
                     'Token Efficiency\n(×1000)', 'Tokens/Step']

    # Create heatmap data
    heatmap_data = stats[['model'] + metrics].set_index('model')

    # Normalize for visualization (0-1 scale, higher is better for all except tokens_per_step)
    normalized = heatmap_data.copy()
    for col in metrics:
        if col == 'tokens_per_step':
            # Inverse: lower is better
            normalized[col] = 1 - (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
        elif col == 'efficiency':
            # Already normalized (higher is better)
            normalized[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
        else:
            # Percentages: scale to 0-1
            normalized[col] = heatmap_data[col] / 100.0

    # Create heatmap
    sns.heatmap(normalized, annot=heatmap_data.round(2), fmt='',
                cmap='RdYlGn', linewidths=0.5, ax=ax,
                xticklabels=metric_labels,
                cbar_kws={'label': 'Performance (normalized)'})

    ax.set_title('Model Performance Heatmap', fontsize=FONT_SIZE + 2, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=FONT_SIZE)
    ax.set_ylabel('Model', fontsize=FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description='Visualize OR-Debug-Bench token tracking results')
    parser.add_argument('--db-dir', type=str, required=True,
                        help='Directory containing SQLite database files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.db_dir}...")
    df = load_all_results(args.db_dir)
    print(f"Loaded {len(df)} records from {df['model'].nunique()} models")

    # Compute statistics
    stats = compute_model_stats(df)
    print("\nModel Statistics:")
    print(stats[['model', 'n', 'rr', 'avg_tokens', 'efficiency']].to_string(index=False))

    # Generate figures
    print("\nGenerating figures...")

    plot_performance_overview(stats, str(output_dir / 'fig1_performance_overview.png'))
    plot_token_efficiency(stats, str(output_dir / 'fig2_token_efficiency.png'))
    plot_rr_at_steps(stats, str(output_dir / 'fig3_rr_at_steps.png'))
    plot_rr_at_tokens(stats, str(output_dir / 'fig4_rr_at_tokens.png'))
    plot_token_composition(stats, str(output_dir / 'fig5_token_composition.png'))
    plot_efficiency_heatmap(stats, str(output_dir / 'fig6_efficiency_heatmap.png'))

    print(f"\nAll figures saved to {output_dir}")


if __name__ == '__main__':
    main()
