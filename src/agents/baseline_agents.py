"""
Baseline agents for OR-Debug-Bench evaluation.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - RandomAgent: Randomly selects valid actions
    - HeuristicAgent: Rule-based debugging strategy

Example:
    >>> from src.agents import HeuristicAgent
    >>> agent = HeuristicAgent()
    >>> action = agent.act(state)
"""

import random
from typing import Optional, List

from src.environments import DebugState, Action, ActionType
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Random baseline agent.

    Selects actions uniformly at random from valid actions.
    Used as a lower-bound baseline for evaluation.
    """

    def __init__(self, seed: Optional[int] = None, name: str = "RandomAgent"):
        """
        Initialize RandomAgent.

        Args:
            seed: Random seed for reproducibility
            name: Agent name
        """
        super().__init__(name=name)
        self._seed = seed
        self._rng = random.Random(seed)

    def act(self, state: DebugState) -> Action:
        """
        Select a random valid action.

        Args:
            state: Current environment state

        Returns:
            Randomly selected action
        """
        valid_actions = self._get_valid_actions(state)
        return self._rng.choice(valid_actions)

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
        # Optionally reset RNG for reproducibility
        if self._seed is not None:
            self._rng = random.Random(self._seed)

    def _get_valid_actions(self, state: DebugState) -> List[Action]:
        """
        Generate list of valid actions for current state.

        Args:
            state: Current environment state

        Returns:
            List of valid Action objects
        """
        actions = []

        # Diagnosis actions
        actions.append(Action(ActionType.GET_IIS))

        for name in state.constraint_names[:10]:  # Limit for efficiency
            actions.append(Action(ActionType.CHECK_SLACK, target=name))

        # Repair actions (on IIS constraints if available)
        targets = state.iis_constraints if state.iis_constraints else state.constraint_names[:5]
        for name in targets:
            actions.append(Action(ActionType.DROP_CONSTRAINT, target=name))

        # Meta actions
        actions.append(Action(ActionType.SUBMIT))

        return actions


class HeuristicAgent(BaseAgent):
    """
    Heuristic baseline agent.

    Follows a simple rule-based strategy:
    1. If no IIS known: GET_IIS
    2. If IIS available: DROP first IIS constraint
    3. If model is optimal: SUBMIT

    This represents a reasonable hand-coded baseline.
    """

    def __init__(self, name: str = "HeuristicAgent"):
        """
        Initialize HeuristicAgent.

        Args:
            name: Agent name
        """
        super().__init__(name=name)
        self._has_iis = False

    def act(self, state: DebugState) -> Action:
        """
        Select action based on heuristic rules.

        Args:
            state: Current environment state

        Returns:
            Selected action
        """
        # If optimal, submit
        if state.is_optimal():
            return Action(ActionType.SUBMIT)

        # If infeasible and no IIS, get IIS
        if state.is_infeasible() and not state.iis_constraints and not self._has_iis:
            self._has_iis = True
            return Action(ActionType.GET_IIS)

        # If IIS available, drop the first constraint
        if state.iis_constraints:
            target = state.iis_constraints[0]
            return Action(ActionType.DROP_CONSTRAINT, target=target)

        # If IIS bounds available, try dropping a related constraint
        if state.iis_bounds:
            # Try to find a constraint that mentions the variable
            for var_name in state.iis_bounds:
                for constr_name in state.constraint_names:
                    if var_name.lower() in constr_name.lower():
                        return Action(ActionType.DROP_CONSTRAINT, target=constr_name)

            # Fallback: drop first constraint
            if state.constraint_names:
                return Action(ActionType.DROP_CONSTRAINT, target=state.constraint_names[0])

        # Fallback: try GET_IIS again or submit
        if state.is_infeasible():
            return Action(ActionType.GET_IIS)

        return Action(ActionType.SUBMIT)

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
        self._has_iis = False


class GreedyDropAgent(BaseAgent):
    """
    Greedy constraint dropping agent.

    Drops IIS constraints one by one until the model becomes feasible.
    More aggressive than HeuristicAgent - drops all IIS constraints
    before checking feasibility.
    """

    def __init__(self, name: str = "GreedyDropAgent"):
        """
        Initialize GreedyDropAgent.

        Args:
            name: Agent name
        """
        super().__init__(name=name)
        self._dropped_constraints: List[str] = []

    def act(self, state: DebugState) -> Action:
        """
        Greedily drop constraints.

        Args:
            state: Current environment state

        Returns:
            Selected action
        """
        # If optimal, submit
        if state.is_optimal():
            return Action(ActionType.SUBMIT)

        # Get IIS if not available
        if state.is_infeasible() and not state.iis_constraints:
            return Action(ActionType.GET_IIS)

        # Drop constraints from IIS that haven't been dropped yet
        for constr in state.iis_constraints:
            if constr not in self._dropped_constraints:
                self._dropped_constraints.append(constr)
                return Action(ActionType.DROP_CONSTRAINT, target=constr)

        # All IIS constraints dropped but still infeasible - get new IIS
        if state.is_infeasible():
            return Action(ActionType.GET_IIS)

        return Action(ActionType.SUBMIT)

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
        self._dropped_constraints = []


class DoNothingAgent(BaseAgent):
    """
    Do-nothing agent for testing.

    Always submits immediately without attempting any repairs.
    Useful as a failure baseline.
    """

    def __init__(self, name: str = "DoNothingAgent"):
        super().__init__(name=name)

    def act(self, state: DebugState) -> Action:
        """Always submit."""
        return Action(ActionType.SUBMIT)

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
