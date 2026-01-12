"""
Reward calculation for OR-Debug-Bench MDP environment.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - RewardCalculator: Computes composite rewards
    - Three reward types: Outcome, Process, Faithfulness

Example:
    >>> from src.environments.reward import RewardCalculator
    >>> calc = RewardCalculator()
    >>> reward = calc.compute_reward(old_state, new_state, action)
"""

from dataclasses import dataclass
from typing import Optional

from .state import DebugState
from .action import Action


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""

    # Outcome rewards (sparse)
    success_reward: float = 100.0  # OPTIMAL achieved
    failure_reward: float = -50.0  # Still INFEASIBLE after submit
    syntax_error_reward: float = -50.0  # Invalid action

    # Process rewards (dense)
    iis_reduction_reward: float = 10.0  # IIS size reduced
    step_penalty: float = -1.0  # Per-step cost
    constraint_preserved_bonus: float = 5.0  # Quality repair

    # Faithfulness penalties
    diagnosis_contradiction_penalty: float = -20.0  # Hallucination
    explanation_inconsistency_penalty: float = -10.0  # Reasoning error


class RewardCalculator:
    """
    Calculates composite rewards for the OR-Debug-Bench environment.

    Rewards are composed of three categories:
    1. Outcome Rewards: Verifiable, sparse rewards for terminal states
    2. Process Rewards: Dense rewards for progress toward solution
    3. Faithfulness Penalties: Penalties for inconsistent reasoning

    The solver acts as the ground-truth oracle (RLVR paradigm).
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize RewardCalculator.

        Args:
            config: Reward configuration. Uses defaults if not provided.
        """
        self.config = config or RewardConfig()

    def compute_reward(
        self,
        old_state: DebugState,
        new_state: DebugState,
        action: Action,
        is_terminal: bool = False,
    ) -> float:
        """
        Compute the total reward for a state transition.

        Args:
            old_state: State before the action
            new_state: State after the action
            action: The action that was taken
            is_terminal: Whether this is a terminal transition

        Returns:
            Total reward (sum of outcome, process, and faithfulness)
        """
        reward = 0.0

        # Outcome reward (only for terminal states)
        if is_terminal:
            reward += self._compute_outcome_reward(new_state, action)

        # Process reward (always computed)
        reward += self._compute_process_reward(old_state, new_state, action)

        # Faithfulness penalty (for diagnosis actions)
        if action.is_diagnosis_action:
            reward += self._compute_faithfulness_penalty(old_state, new_state, action)

        return reward

    def _compute_outcome_reward(self, state: DebugState, action: Action) -> float:
        """
        Compute outcome reward for terminal states.

        Args:
            state: Final state
            action: Final action (e.g., SUBMIT)

        Returns:
            Outcome reward
        """
        if state.is_optimal():
            return self.config.success_reward
        elif state.is_infeasible():
            return self.config.failure_reward
        else:
            # Unbounded or other status
            return self.config.failure_reward

    def _compute_process_reward(
        self,
        old_state: DebugState,
        new_state: DebugState,
        action: Action,
    ) -> float:
        """
        Compute process reward for dense credit assignment.

        Args:
            old_state: State before action
            new_state: State after action
            action: Action taken

        Returns:
            Process reward
        """
        reward = 0.0

        # Step penalty (always applied)
        reward += self.config.step_penalty

        # IIS reduction bonus
        if action.is_repair_action:
            old_iis_size = old_state.get_iis_size()
            new_iis_size = new_state.get_iis_size()

            if new_iis_size < old_iis_size:
                # Reward proportional to reduction
                reduction = old_iis_size - new_iis_size
                reward += self.config.iis_reduction_reward * reduction

        # Constraint preservation bonus (for non-drop repairs)
        if action.is_repair_action and action.action_type.value != "drop_constraint":
            # Bonus for preserving constraint count
            if new_state.get_constraint_count() >= old_state.get_constraint_count():
                reward += self.config.constraint_preserved_bonus

        return reward

    def _compute_faithfulness_penalty(
        self,
        old_state: DebugState,
        new_state: DebugState,
        action: Action,
    ) -> float:
        """
        Compute faithfulness penalty for diagnosis actions.

        Penalizes hallucination and inconsistent reasoning.

        Args:
            old_state: State before action
            new_state: State after action
            action: Diagnosis action taken

        Returns:
            Faithfulness penalty (negative or zero)
        """
        # Currently a placeholder - would require tracking agent explanations
        # to implement full faithfulness checking
        return 0.0

    def compute_episode_return(self, rewards: list) -> float:
        """
        Compute total return for an episode.

        Args:
            rewards: List of rewards from each step

        Returns:
            Total episode return
        """
        return sum(rewards)

    def get_reward_breakdown(
        self,
        old_state: DebugState,
        new_state: DebugState,
        action: Action,
        is_terminal: bool = False,
    ) -> dict:
        """
        Get detailed breakdown of reward components.

        Args:
            old_state: State before action
            new_state: State after action
            action: Action taken
            is_terminal: Whether terminal state

        Returns:
            Dictionary with individual reward components
        """
        outcome = 0.0
        if is_terminal:
            outcome = self._compute_outcome_reward(new_state, action)

        process = self._compute_process_reward(old_state, new_state, action)
        faithfulness = self._compute_faithfulness_penalty(old_state, new_state, action)

        return {
            "outcome": outcome,
            "process": process,
            "faithfulness": faithfulness,
            "total": outcome + process + faithfulness,
        }
