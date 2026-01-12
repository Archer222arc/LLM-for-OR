"""
Episode statistics helpers for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A4_Evaluation_Metrics.md

Key Components:
    - EpisodeTracker: Track statistics during an episode
    - aggregate_trajectories: Aggregate trajectory data

Example:
    >>> from src.evaluation import EpisodeTracker
    >>> tracker = EpisodeTracker()
    >>> tracker.record_step(state, action, reward)
    >>> result = tracker.finalize(success=True)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.environments import DebugState, Action, ActionType
from .metrics import EpisodeResult


@dataclass
class EpisodeTracker:
    """
    Track statistics during an episode.

    Records step-by-step data and computes the final EpisodeResult.

    Attributes:
        agent_name: Name of the agent being tracked
        problem_id: Identifier for the problem instance
        ground_truth_fix: The known correct fix
        ground_truth_iis: The actual IIS constraints from dataset
        original_objective: Original objective value (for OP calculation)
        original_constraints: Original constraint names (for FP calculation)
    """

    agent_name: str = ""
    problem_id: str = ""
    ground_truth_fix: Optional[str] = None
    ground_truth_iis: List[str] = field(default_factory=list)
    original_objective: Optional[float] = None
    original_constraints: List[str] = field(default_factory=list)

    # Internal tracking
    _steps: int = field(default=0, init=False)
    _total_reward: float = field(default=0.0, init=False)
    _trajectory: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _iis_actions: List[str] = field(default_factory=list, init=False)
    _diagnosed_constraints: List[str] = field(default_factory=list, init=False)
    _initial_status: str = field(default="", init=False)
    _final_status: str = field(default="", init=False)
    _success_at_step: Optional[int] = field(default=None, init=False)
    _recovered_objective: Optional[float] = field(default=None, init=False)
    _remaining_constraints: List[str] = field(default_factory=list, init=False)

    # Test-Time Compute tracking (set externally from agent)
    _total_tokens: int = field(default=0, init=False)
    _total_input_tokens: int = field(default=0, init=False)
    _total_output_tokens: int = field(default=0, init=False)
    _total_reasoning_tokens: int = field(default=0, init=False)
    _api_call_count: int = field(default=0, init=False)
    _tokens_per_step: List[int] = field(default_factory=list, init=False)
    _api_calls: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _wall_clock_seconds: float = field(default=0.0, init=False)

    def reset(self) -> None:
        """Reset tracker for a new episode."""
        self._steps = 0
        self._total_reward = 0.0
        self._trajectory = []
        self._iis_actions = []
        self._diagnosed_constraints = []
        self._initial_status = ""
        self._final_status = ""
        self._success_at_step = None
        self._recovered_objective = None
        self._remaining_constraints = []
        # Reset token tracking
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_tokens = 0
        self._api_call_count = 0
        self._tokens_per_step = []
        self._api_calls = []
        self._wall_clock_seconds = 0.0

    def record_step(
        self,
        state: DebugState,
        action: Action,
        reward: float,
        next_state: DebugState,
        done: bool,
    ) -> None:
        """
        Record a single step.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode is done
        """
        self._steps += 1
        self._total_reward += reward

        # Track initial status
        if self._steps == 1:
            self._initial_status = state.solver_status

        # Track final status
        self._final_status = next_state.solver_status

        # Track success_at_step (first step when OPTIMAL is achieved)
        if self._success_at_step is None and next_state.is_optimal():
            self._success_at_step = self._steps
            # Record objective and constraints at success
            self._recovered_objective = next_state.objective
            self._remaining_constraints = list(next_state.constraint_names) if next_state.constraint_names else []

        # Record trajectory
        self._trajectory.append({
            "step": self._steps,
            "state_status": state.solver_status,
            "action_type": action.action_type.value,
            "action_target": action.target,
            "reward": reward,
            "next_status": next_state.solver_status,
            "done": done,
        })

        # Track diagnosed constraints (all constraints agent targets)
        if action.action_type == ActionType.DROP_CONSTRAINT and action.target:
            if action.target not in self._diagnosed_constraints:
                self._diagnosed_constraints.append(action.target)
            # Also track if this was a correct IIS identification
            if state.iis_constraints and action.target in state.iis_constraints:
                self._iis_actions.append(action.target)

    def finalize(
        self,
        success: bool,
        token_stats: Optional[Dict[str, Any]] = None,
        elapsed_seconds: float = 0.0,
    ) -> EpisodeResult:
        """
        Finalize and return the episode result.

        Args:
            success: Whether the episode was successful
            token_stats: Optional token statistics from agent.get_token_stats()
            elapsed_seconds: Wall clock time for the episode

        Returns:
            EpisodeResult containing all recorded data including token usage
        """
        # Populate token fields from agent stats if provided
        if token_stats:
            self._total_tokens = token_stats.get('total_tokens', 0)
            self._total_input_tokens = token_stats.get('total_input_tokens', 0)
            self._total_output_tokens = token_stats.get('total_output_tokens', 0)
            self._total_reasoning_tokens = token_stats.get('total_reasoning_tokens', 0)
            self._api_call_count = token_stats.get('api_call_count', 0)
            self._tokens_per_step = token_stats.get('tokens_per_step', [])
            self._api_calls = token_stats.get('api_calls', [])

        self._wall_clock_seconds = elapsed_seconds

        return EpisodeResult(
            success=success,
            final_status=self._final_status,
            steps=self._steps,
            total_reward=self._total_reward,
            trajectory=self._trajectory.copy(),
            iis_actions=self._iis_actions.copy(),
            diagnosed_constraints=self._diagnosed_constraints.copy(),
            ground_truth_iis=list(self.ground_truth_iis) if self.ground_truth_iis else [],
            ground_truth_fix=self.ground_truth_fix,
            agent_name=self.agent_name,
            problem_id=self.problem_id,
            # OP/FP/RR@k fields
            original_objective=self.original_objective,
            recovered_objective=self._recovered_objective,
            original_constraint_count=len(self.original_constraints) if self.original_constraints else 0,
            remaining_constraint_count=len(self._remaining_constraints),
            original_constraints=list(self.original_constraints) if self.original_constraints else [],
            remaining_constraints=self._remaining_constraints.copy(),
            success_at_step=self._success_at_step,
            # Test-Time Compute fields
            total_tokens=self._total_tokens,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_reasoning_tokens=self._total_reasoning_tokens,
            api_call_count=self._api_call_count,
            tokens_per_step=list(self._tokens_per_step),
            api_calls=list(self._api_calls),
            wall_clock_seconds=self._wall_clock_seconds,
        )


def aggregate_trajectories(
    results: List[EpisodeResult],
) -> Dict[str, Any]:
    """
    Aggregate trajectory data across multiple episodes.

    Args:
        results: List of episode results

    Returns:
        Aggregated statistics about trajectories
    """
    if not results:
        return {}

    all_actions = []
    action_counts: Dict[str, int] = {}

    for result in results:
        for step in result.trajectory:
            action_type = step.get("action_type", "unknown")
            all_actions.append(action_type)
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

    total_actions = len(all_actions)

    return {
        "total_actions": total_actions,
        "action_counts": action_counts,
        "action_proportions": {
            k: v / total_actions for k, v in action_counts.items()
        } if total_actions > 0 else {},
        "avg_trajectory_length": total_actions / len(results) if results else 0,
    }


def extract_action_sequence(result: EpisodeResult) -> List[str]:
    """
    Extract action sequence from an episode result.

    Args:
        result: Episode result

    Returns:
        List of action type strings in order
    """
    return [step.get("action_type", "unknown") for step in result.trajectory]


def compute_action_diversity(results: List[EpisodeResult]) -> float:
    """
    Compute action diversity across episodes.

    Measures how diverse the action sequences are using
    unique action types per episode.

    Args:
        results: List of episode results

    Returns:
        Average number of unique action types per episode
    """
    if not results:
        return 0.0

    diversities = []
    for result in results:
        action_types = set(extract_action_sequence(result))
        diversities.append(len(action_types))

    return sum(diversities) / len(diversities)
