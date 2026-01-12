"""
MDP Environment for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - SolverDebugEnv: Gymnasium-style MDP environment
    - DebugState: State representation
    - Action/ActionType: Action space definitions
    - RewardCalculator: Reward computation

Example:
    >>> from src.solvers import GurobiSolver
    >>> from src.environments import SolverDebugEnv, Action, ActionType
    >>> solver = GurobiSolver.from_file("model.mps")
    >>> env = SolverDebugEnv(solver, problem_nl="Minimize cost...")
    >>> state, info = env.reset()
    >>> action = Action(ActionType.GET_IIS)
    >>> state, reward, done, truncated, info = env.step(action)
"""

# Action definitions
from .action import (
    ActionType,
    Action,
    get_iis,
    check_slack,
    relax_constraint,
    drop_constraint,
    update_rhs,
    update_bounds,
    reset,
    submit,
)

# State definitions
from .state import DebugState, StepResult

# Reward calculation
from .reward import RewardCalculator, RewardConfig

# Main environment
from .solver_gym import SolverDebugEnv

__all__ = [
    # Actions
    "ActionType",
    "Action",
    "get_iis",
    "check_slack",
    "relax_constraint",
    "drop_constraint",
    "update_rhs",
    "update_bounds",
    "reset",
    "submit",
    # State
    "DebugState",
    "StepResult",
    # Reward
    "RewardCalculator",
    "RewardConfig",
    # Environment
    "SolverDebugEnv",
]
