"""
Agent implementations for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - BaseAgent: Abstract base class for all agents
    - LLMAgent: LLM-based debugging agent
    - RandomAgent: Random baseline agent
    - HeuristicAgent: Rule-based baseline agent

Example:
    >>> from src.environments import SolverDebugEnv
    >>> from src.agents import HeuristicAgent
    >>> agent = HeuristicAgent()
    >>> state, _ = env.reset()
    >>> action = agent.act(state)
"""

# Base class
from .base_agent import BaseAgent

# Baseline agents
from .baseline_agents import (
    RandomAgent,
    HeuristicAgent,
    GreedyDropAgent,
    DoNothingAgent,
)

# LLM agents
from .llm_agent import LLMAgent, MockLLMAgent

# Prompts
from .prompts import (
    SYSTEM_PROMPT,
    format_state,
    format_history,
    FEW_SHOT_EXAMPLES,
    format_few_shot_prompt,
    get_system_prompt,
)

__all__ = [
    # Base
    "BaseAgent",
    # Baselines
    "RandomAgent",
    "HeuristicAgent",
    "GreedyDropAgent",
    "DoNothingAgent",
    # LLM
    "LLMAgent",
    "MockLLMAgent",
    # Prompts
    "SYSTEM_PROMPT",
    "format_state",
    "format_history",
    "FEW_SHOT_EXAMPLES",
    "format_few_shot_prompt",
    "get_system_prompt",
]
