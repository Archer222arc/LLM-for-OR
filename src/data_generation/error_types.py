"""
Error type definitions for Saboteur Agent.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A3_Data_Generation.md

Key Components:
    - ErrorType: Enum for error categories (A-D)
    - InjectionResult: Result of error injection operation

Example:
    >>> from src.data_generation.error_types import ErrorType, InjectionResult
    >>> result = InjectionResult(
    ...     success=True,
    ...     error_type=ErrorType.TYPE_A,
    ...     target_name="constraint_1",
    ...     original_value="<=",
    ...     modified_value=">=",
    ...     solver_status="INFEASIBLE",
    ...     ground_truth_fix="Change constraint_1 from >= back to <="
    ... )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List


class ErrorType(Enum):
    """
    Classification of error types for OR-Debug-Bench.

    Type A: Bound Error - Flipping constraint direction (≤ ↔ ≥)
    Type B: Variable Error - Changing variable type (INTEGER ↔ CONTINUOUS)
    Type C: Logic Error - Removing terms from constraint expressions
    Type D: Conflict Error - Adding contradicting constraints
    Type E: Multi-Constraint Conflict - Requires fixing 2+ constraints
    Type F: Hidden Dependency - Root cause not directly in IIS
    """

    TYPE_A = "A"  # Bound Error: Flip constraint sense
    TYPE_B = "B"  # Variable Error: Change variable type
    TYPE_C = "C"  # Logic Error: Remove expression term
    TYPE_D = "D"  # Conflict Error: Add contradicting constraint
    TYPE_E = "E"  # Multi-Constraint: Multiple fixes required
    TYPE_F = "F"  # Hidden Dependency: Indirect root cause

    @property
    def description(self) -> str:
        """Human-readable description of the error type."""
        descriptions = {
            "A": "Bound Error - Constraint direction flip (≤ ↔ ≥)",
            "B": "Variable Error - Variable type modification (INTEGER ↔ CONTINUOUS)",
            "C": "Logic Error - Expression term removal",
            "D": "Conflict Error - Contradicting constraint addition",
            "E": "Multi-Constraint Conflict - Requires fixing 2+ constraints",
            "F": "Hidden Dependency - Root cause not directly visible in IIS",
        }
        return descriptions[self.value]


class Difficulty(Enum):
    """Difficulty classification based on IIS size."""

    EASY = "easy"      # IIS size 1-2 constraints
    MEDIUM = "medium"  # IIS size 3-5 constraints
    HARD = "hard"      # IIS size 6+ constraints

    @classmethod
    def from_iis_size(cls, iis_size: int) -> "Difficulty":
        """Classify difficulty based on IIS size."""
        if iis_size <= 2:
            return cls.EASY
        elif iis_size <= 5:
            return cls.MEDIUM
        else:
            return cls.HARD


@dataclass
class InjectionResult:
    """
    Result of an error injection operation.

    Attributes:
        success: Whether the injection successfully caused infeasibility/unboundedness
        error_type: The type of error injected (A/B/C/D)
        target_name: Name of the constraint/variable that was modified
        original_value: The original value before modification
        modified_value: The value after modification
        solver_status: Solver status after injection (INFEASIBLE/UNBOUNDED/etc.)
        ground_truth_fix: Description of how to fix the error
        metadata: Additional information about the injection
        difficulty: Difficulty classification (easy/medium/hard)
        iis_size: Number of constraints/bounds in IIS
        iis_constraints: List of constraint names in IIS
        iis_bounds: List of variable names with bounds in IIS
        original_objective: Objective value of original feasible model
    """

    success: bool
    error_type: ErrorType
    target_name: str
    original_value: Any
    modified_value: Any
    solver_status: str
    ground_truth_fix: str
    metadata: Optional[dict] = None
    # New fields for enhanced tracking
    difficulty: Difficulty = Difficulty.MEDIUM
    iis_size: int = 0
    iis_constraints: List[str] = field(default_factory=list)
    iis_bounds: List[str] = field(default_factory=list)
    original_objective: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "error_type": self.error_type.value,
            "target_name": self.target_name,
            "original_value": str(self.original_value),
            "modified_value": str(self.modified_value),
            "solver_status": self.solver_status,
            "ground_truth_fix": self.ground_truth_fix,
            "metadata": self.metadata or {},
            "difficulty": self.difficulty.value,
            "iis_size": self.iis_size,
            "iis_constraints": self.iis_constraints,
            "iis_bounds": self.iis_bounds,
            "original_objective": self.original_objective,
        }

    @property
    def is_infeasible(self) -> bool:
        """Check if injection resulted in infeasibility."""
        return self.solver_status in ["INFEASIBLE", "INF_OR_UNBD"]

    @property
    def is_unbounded(self) -> bool:
        """Check if injection resulted in unboundedness."""
        return self.solver_status in ["UNBOUNDED", "INF_OR_UNBD"]
