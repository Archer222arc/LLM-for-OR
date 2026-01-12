"""
Abstract base class for optimization solvers.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/

Key Components:
    - BaseSolver: Abstract interface for solver implementations
    - SolverState: Unified state representation
    - IISResult: Infeasibility diagnosis result

Example:
    >>> class MySolver(BaseSolver):
    ...     def solve(self) -> SolverState:
    ...         # Implementation
    ...         pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SolverState:
    """Represents the state of a solver after optimization."""

    status: str  # OPTIMAL, INFEASIBLE, UNBOUNDED, INF_OR_UNBD, SUBOPTIMAL, ERROR
    objective: Optional[float] = None  # Objective value (if OPTIMAL)
    gap: Optional[float] = None  # MIP gap (for MIP problems)
    solve_time: float = 0.0  # Time spent solving (seconds)
    node_count: int = 0  # Branch-and-bound nodes explored
    iteration_count: int = 0  # Simplex iterations


@dataclass
class IISResult:
    """Represents an Irreducible Infeasible Subsystem."""

    constraints: List[str] = field(default_factory=list)  # Conflicting constraint names
    bounds: List[str] = field(default_factory=list)  # Variables with bound conflicts

    @property
    def size(self) -> int:
        """Total size of the IIS."""
        return len(self.constraints) + len(self.bounds)

    @property
    def is_empty(self) -> bool:
        """Check if IIS is empty."""
        return self.size == 0


@dataclass
class ConstraintInfo:
    """Information about a single constraint."""

    name: str
    sense: str  # '<', '>', '='
    rhs: float
    slack: Optional[float] = None  # Slack value (if solved)
    is_in_iis: bool = False  # Whether this constraint is in the IIS


@dataclass
class VariableInfo:
    """Information about a single variable."""

    name: str
    vtype: str  # 'C' (continuous), 'B' (binary), 'I' (integer)
    lb: float  # Lower bound
    ub: float  # Upper bound
    value: Optional[float] = None  # Solution value (if solved)
    is_lb_in_iis: bool = False  # Lower bound in IIS
    is_ub_in_iis: bool = False  # Upper bound in IIS


class BaseSolver(ABC):
    """
    Abstract base class for optimization solvers.

    Defines the interface that all solver implementations must follow
    for compatibility with the OR-Debug-Bench environment.
    """

    @abstractmethod
    def solve(self) -> SolverState:
        """
        Solve the optimization model.

        Returns:
            SolverState with status, objective, and other metrics.
        """
        pass

    @abstractmethod
    def compute_iis(self) -> IISResult:
        """
        Compute the Irreducible Infeasible Subsystem.

        Should only be called when model status is INFEASIBLE.

        Returns:
            IISResult containing conflicting constraints and bounds.
        """
        pass

    @abstractmethod
    def get_constraint_info(self, name: str) -> ConstraintInfo:
        """Get information about a specific constraint."""
        pass

    @abstractmethod
    def get_variable_info(self, name: str) -> VariableInfo:
        """Get information about a specific variable."""
        pass

    @abstractmethod
    def clone(self) -> 'BaseSolver':
        """Create an independent copy of the solver and model."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the model to its original state."""
        pass

    @abstractmethod
    def get_all_constraints(self) -> List[str]:
        """Get names of all constraints in the model."""
        pass

    @abstractmethod
    def get_all_variables(self) -> List[str]:
        """Get names of all variables in the model."""
        pass
