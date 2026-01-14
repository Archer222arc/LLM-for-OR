"""
GRPO Training module for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/05_TRAINING.md

Key Components:
    - gurobi_rewards: Custom reward functions using Gurobi solver
    - action_parser: Parse actions from LLM completions

Example:
    >>> from src.training import gurobi_reward_func
    >>> rewards = gurobi_reward_func(completions, problem_ids, model_files)
"""

from .gurobi_rewards import gurobi_reward_func
from .action_parser import ActionParser

__all__ = [
    "gurobi_reward_func",
    "ActionParser",
]
