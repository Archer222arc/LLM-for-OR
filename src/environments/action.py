"""
Action definitions for OR-Debug-Bench MDP environment.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - ActionType: Enum for available actions
    - Action: Data class for action representation

Example:
    >>> from src.environments.action import ActionType, Action
    >>> action = Action(ActionType.DROP_CONSTRAINT, target="constraint_1")
    >>> print(action.is_repair_action)
    True
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any


class ActionType(Enum):
    """
    Classification of actions in the OR-Debug-Bench environment.

    Actions are organized into three categories:
    - Diagnosis: Query solver for information (GET_IIS, CHECK_SLACK)
    - Repair: Modify the model (RELAX_CONSTRAINT, DROP_CONSTRAINT, UPDATE_RHS)
    - Meta: Control episode flow (RESET, SUBMIT)
    """

    # Diagnosis Actions
    GET_IIS = "get_iis"
    CHECK_SLACK = "check_slack"

    # Repair Actions
    RELAX_CONSTRAINT = "relax_constraint"
    DROP_CONSTRAINT = "drop_constraint"
    UPDATE_RHS = "update_rhs"
    UPDATE_BOUNDS = "update_bounds"

    # Meta Actions
    RESET = "reset"
    SUBMIT = "submit"

    @property
    def is_diagnosis(self) -> bool:
        """Check if this is a diagnosis action."""
        return self in [ActionType.GET_IIS, ActionType.CHECK_SLACK]

    @property
    def is_repair(self) -> bool:
        """Check if this is a repair action."""
        return self in [
            ActionType.RELAX_CONSTRAINT,
            ActionType.DROP_CONSTRAINT,
            ActionType.UPDATE_RHS,
            ActionType.UPDATE_BOUNDS,
        ]

    @property
    def is_meta(self) -> bool:
        """Check if this is a meta action."""
        return self in [ActionType.RESET, ActionType.SUBMIT]

    @property
    def requires_target(self) -> bool:
        """Check if this action requires a target (constraint/variable name)."""
        return self in [
            ActionType.CHECK_SLACK,
            ActionType.RELAX_CONSTRAINT,
            ActionType.DROP_CONSTRAINT,
            ActionType.UPDATE_RHS,
            ActionType.UPDATE_BOUNDS,
        ]

    @property
    def requires_value(self) -> bool:
        """Check if this action requires a value parameter."""
        return self in [
            ActionType.RELAX_CONSTRAINT,
            ActionType.UPDATE_RHS,
            ActionType.UPDATE_BOUNDS,
        ]


@dataclass
class Action:
    """
    Represents an action in the OR-Debug-Bench environment.

    Attributes:
        action_type: The type of action to perform
        target: Name of constraint/variable to act on (if required)
        value: Numerical value for the action (if required)
        metadata: Additional action-specific data
    """

    action_type: ActionType
    target: Optional[str] = None
    value: Optional[float] = None
    value2: Optional[float] = None  # For UPDATE_BOUNDS (lb, ub)
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate action parameters."""
        if self.action_type.requires_target and self.target is None:
            raise ValueError(
                f"Action {self.action_type.value} requires a target parameter"
            )
        if self.action_type.requires_value and self.value is None:
            raise ValueError(
                f"Action {self.action_type.value} requires a value parameter"
            )

    @property
    def is_diagnosis_action(self) -> bool:
        """Check if this is a diagnosis action."""
        return self.action_type.is_diagnosis

    @property
    def is_repair_action(self) -> bool:
        """Check if this is a repair action."""
        return self.action_type.is_repair

    @property
    def is_meta_action(self) -> bool:
        """Check if this is a meta action."""
        return self.action_type.is_meta

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action_type": self.action_type.value,
            "target": self.target,
            "value": self.value,
            "value2": self.value2,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        """Create Action from dictionary."""
        return cls(
            action_type=ActionType(data["action_type"]),
            target=data.get("target"),
            value=data.get("value"),
            value2=data.get("value2"),
            metadata=data.get("metadata"),
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.target and self.value is not None:
            return f"{self.action_type.value}({self.target}, {self.value})"
        elif self.target:
            return f"{self.action_type.value}({self.target})"
        else:
            return f"{self.action_type.value}()"


# Convenience factory functions
def get_iis() -> Action:
    """Create GET_IIS action."""
    return Action(ActionType.GET_IIS)


def check_slack(constraint_name: str) -> Action:
    """Create CHECK_SLACK action."""
    return Action(ActionType.CHECK_SLACK, target=constraint_name)


def relax_constraint(constraint_name: str, epsilon: float) -> Action:
    """Create RELAX_CONSTRAINT action."""
    return Action(ActionType.RELAX_CONSTRAINT, target=constraint_name, value=epsilon)


def drop_constraint(constraint_name: str) -> Action:
    """Create DROP_CONSTRAINT action."""
    return Action(ActionType.DROP_CONSTRAINT, target=constraint_name)


def update_rhs(constraint_name: str, new_rhs: float) -> Action:
    """Create UPDATE_RHS action."""
    return Action(ActionType.UPDATE_RHS, target=constraint_name, value=new_rhs)


def update_bounds(variable_name: str, lb: float, ub: float) -> Action:
    """Create UPDATE_BOUNDS action."""
    return Action(ActionType.UPDATE_BOUNDS, target=variable_name, value=lb, value2=ub)


def reset() -> Action:
    """Create RESET action."""
    return Action(ActionType.RESET)


def submit() -> Action:
    """Create SUBMIT action."""
    return Action(ActionType.SUBMIT)
