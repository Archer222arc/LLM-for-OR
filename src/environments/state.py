"""
State representation for OR-Debug-Bench MDP environment.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - DebugState: Complete state representation for the MDP

Example:
    >>> from src.environments.state import DebugState
    >>> state = DebugState(
    ...     problem_nl="Minimize cost...",
    ...     solver_status="INFEASIBLE",
    ...     iis_constraints=["c1", "c2"],
    ... )
    >>> print(state.get_iis_size())
    2
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DebugState:
    """
    Represents the state in the OR-Debug-Bench MDP environment.

    State = (problem_nl, solver_status, iis_log, model_info, history)

    Attributes:
        problem_nl: Natural language problem description
        solver_status: Current solver status (OPTIMAL/INFEASIBLE/UNBOUNDED/etc.)
        iis_constraints: List of constraint names in the IIS
        iis_bounds: List of variable names with bound conflicts in IIS
        constraint_names: All constraint names in the model
        variable_names: All variable names in the model
        constraint_info: Detailed info for each constraint
        variable_info: Detailed info for each variable
        history: List of previous (action, result) pairs
        step_count: Number of steps taken in this episode
        objective: Current objective value (if OPTIMAL)
    """

    # Core state components
    problem_nl: str = ""
    solver_status: str = "LOADED"

    # IIS information
    iis_constraints: List[str] = field(default_factory=list)
    iis_bounds: List[str] = field(default_factory=list)

    # Model structure
    constraint_names: List[str] = field(default_factory=list)
    variable_names: List[str] = field(default_factory=list)

    # Detailed info (optional, populated on demand)
    constraint_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    variable_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # History tracking
    history: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0

    # Solution info
    objective: Optional[float] = None
    gap: Optional[float] = None

    def get_iis_size(self) -> int:
        """Get total size of the IIS."""
        return len(self.iis_constraints) + len(self.iis_bounds)

    def is_infeasible(self) -> bool:
        """Check if the model is currently infeasible."""
        return self.solver_status in ["INFEASIBLE", "INF_OR_UNBD"]

    def is_optimal(self) -> bool:
        """Check if the model is currently optimal."""
        return self.solver_status == "OPTIMAL"

    def is_unbounded(self) -> bool:
        """Check if the model is currently unbounded."""
        return self.solver_status in ["UNBOUNDED", "INF_OR_UNBD"]

    def has_iis(self) -> bool:
        """Check if IIS information is available."""
        return len(self.iis_constraints) > 0 or len(self.iis_bounds) > 0

    def get_constraint_count(self) -> int:
        """Get number of constraints in the model."""
        return len(self.constraint_names)

    def get_variable_count(self) -> int:
        """Get number of variables in the model."""
        return len(self.variable_names)

    def add_to_history(self, action_dict: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Add an action-result pair to history."""
        self.history.append({
            "step": self.step_count,
            "action": action_dict,
            "result": result,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "problem_nl": self.problem_nl,
            "solver_status": self.solver_status,
            "iis_constraints": self.iis_constraints.copy(),
            "iis_bounds": self.iis_bounds.copy(),
            "constraint_names": self.constraint_names.copy(),
            "variable_names": self.variable_names.copy(),
            "constraint_info": self.constraint_info.copy(),
            "variable_info": self.variable_info.copy(),
            "history": [h.copy() for h in self.history],
            "step_count": self.step_count,
            "objective": self.objective,
            "gap": self.gap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebugState":
        """Create DebugState from dictionary."""
        return cls(
            problem_nl=data.get("problem_nl", ""),
            solver_status=data.get("solver_status", "LOADED"),
            iis_constraints=data.get("iis_constraints", []),
            iis_bounds=data.get("iis_bounds", []),
            constraint_names=data.get("constraint_names", []),
            variable_names=data.get("variable_names", []),
            constraint_info=data.get("constraint_info", {}),
            variable_info=data.get("variable_info", {}),
            history=data.get("history", []),
            step_count=data.get("step_count", 0),
            objective=data.get("objective"),
            gap=data.get("gap"),
        )

    def copy(self) -> "DebugState":
        """Create a deep copy of this state."""
        return DebugState.from_dict(self.to_dict())

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"DebugState(status={self.solver_status}, step={self.step_count})",
            f"  Constraints: {self.get_constraint_count()}",
            f"  Variables: {self.get_variable_count()}",
            f"  IIS size: {self.get_iis_size()}",
        ]
        if self.objective is not None:
            lines.append(f"  Objective: {self.objective}")
        if self.iis_constraints:
            lines.append(f"  IIS constraints: {self.iis_constraints[:5]}...")
        return "\n".join(lines)


@dataclass
class StepResult:
    """
    Result of a single step in the environment.

    Attributes:
        state: New state after the action
        reward: Reward received
        terminated: Whether the episode ended (success or failure)
        truncated: Whether the episode was cut short (max steps)
        info: Additional information about the step
    """

    state: DebugState
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self):
        """Convert to Gymnasium-style tuple."""
        return (self.state, self.reward, self.terminated, self.truncated, self.info)
