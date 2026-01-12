"""
Base agent class for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - BaseAgent: Abstract base class for all agents

Example:
    >>> from src.agents import BaseAgent
    >>> class MyAgent(BaseAgent):
    ...     def act(self, state): ...
    ...     def reset(self): ...
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from src.environments import DebugState, Action


class BaseAgent(ABC):
    """
    Abstract base class for all debugging agents.

    Agents interact with the SolverDebugEnv by observing DebugState
    and producing Action objects. This class defines the common interface
    that all agents must implement.

    Subclasses must implement:
        - act(state): Choose an action given the current state
        - reset(): Reset agent's internal state for a new episode
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            name: Optional name for the agent. Defaults to class name.
        """
        self._name = name or self.__class__.__name__
        self._episode_history: List[Dict[str, Any]] = []

    @abstractmethod
    def act(self, state: DebugState) -> Action:
        """
        Choose an action given the current state.

        Args:
            state: Current environment state

        Returns:
            Action to take
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset agent's internal state for a new episode.

        Called at the beginning of each episode before the first act() call.
        """
        pass

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self._name

    def record_step(
        self,
        state: DebugState,
        action: Action,
        reward: float,
        next_state: DebugState,
        done: bool,
    ) -> None:
        """
        Record a step for learning or analysis.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
        """
        self._episode_history.append({
            "state": state.to_dict(),
            "action": action.to_dict(),
            "reward": reward,
            "next_state": next_state.to_dict(),
            "done": done,
        })

    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Get the recorded history for the current episode."""
        return self._episode_history.copy()

    def clear_history(self) -> None:
        """Clear the episode history."""
        self._episode_history = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}')"
