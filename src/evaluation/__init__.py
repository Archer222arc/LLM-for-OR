"""
Evaluation module for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A4_Evaluation_Metrics.md

Key Components:
    - EpisodeResult: Container for episode outcome data
    - MetricsCalculator: Compute evaluation metrics
    - BenchmarkRunner: Run benchmarks and collect results
    - EpisodeTracker: Track statistics during episodes

Example:
    >>> from src.evaluation import BenchmarkRunner, MetricsCalculator
    >>> runner = BenchmarkRunner()
    >>> results = runner.run_benchmark(problems, agent)
    >>> summary = runner.get_summary()
    >>> print(runner.format_summary())
"""

# Core metrics
from .metrics import (
    TokenUsage,
    EpisodeResult,
    BenchmarkConfig,
    MetricsCalculator,
)

# Episode tracking
from .episode_stats import (
    EpisodeTracker,
    aggregate_trajectories,
    extract_action_sequence,
    compute_action_diversity,
)

# Benchmark running
from .benchmark_runner import (
    BenchmarkProblem,
    BenchmarkRunner,
)

# Result storage
from .result_db import ResultDB

__all__ = [
    # Metrics
    "TokenUsage",
    "EpisodeResult",
    "BenchmarkConfig",
    "MetricsCalculator",
    # Episode tracking
    "EpisodeTracker",
    "aggregate_trajectories",
    "extract_action_sequence",
    "compute_action_diversity",
    # Benchmark
    "BenchmarkProblem",
    "BenchmarkRunner",
    # Storage
    "ResultDB",
]
