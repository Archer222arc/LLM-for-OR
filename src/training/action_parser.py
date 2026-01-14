"""
Action parser for extracting actions from LLM completions.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/05_TRAINING.md

Key Components:
    - ActionParser: Parse action strings from LLM completions
    - extract_diagnosis: Extract diagnosis from <think> tags

Example:
    >>> from src.training.action_parser import ActionParser
    >>> action = ActionParser.parse("Action: RELAX_CONSTRAINT(c1, 5.0)")
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ParsedAction:
    """Parsed action from LLM completion."""
    action_type: str
    target: Optional[str] = None
    value: Optional[float] = None
    value2: Optional[float] = None
    raw_text: str = ""
    is_valid: bool = True
    error_message: Optional[str] = None


class ActionParser:
    """
    Parser for extracting actions from LLM-generated completions.

    Supports formats:
    - Action: ACTION_TYPE(args)
    - Action: ACTION_TYPE
    """

    # Action pattern: Action: ACTION_TYPE(arg1, arg2, ...)
    ACTION_PATTERN = re.compile(
        r"Action:\s*(\w+)\s*(?:\(([^)]*)\))?",
        re.IGNORECASE
    )

    # Think tag pattern for diagnosis extraction
    THINK_PATTERN = re.compile(
        r"<think>(.*?)</think>",
        re.DOTALL | re.IGNORECASE
    )

    # Valid action types
    VALID_ACTIONS = {
        "GET_IIS", "CHECK_SLACK",
        "RELAX_CONSTRAINT", "DROP_CONSTRAINT",
        "UPDATE_RHS", "UPDATE_BOUNDS",
        "RESET", "SUBMIT"
    }

    @classmethod
    def parse(cls, completion: str) -> ParsedAction:
        """
        Parse an action from LLM completion text.

        Args:
            completion: Raw completion text from LLM

        Returns:
            ParsedAction with parsed components
        """
        # Find action in completion
        match = cls.ACTION_PATTERN.search(completion)

        if not match:
            return ParsedAction(
                action_type="UNKNOWN",
                raw_text=completion[:200],
                is_valid=False,
                error_message="No action found in completion"
            )

        action_type = match.group(1).upper()
        args_str = match.group(2) if match.group(2) else ""

        # Validate action type
        if action_type not in cls.VALID_ACTIONS:
            return ParsedAction(
                action_type=action_type,
                raw_text=match.group(0),
                is_valid=False,
                error_message=f"Invalid action type: {action_type}"
            )

        # Parse arguments
        target, value, value2 = cls._parse_args(args_str)

        # Validate required arguments
        is_valid, error = cls._validate_args(action_type, target, value, value2)

        return ParsedAction(
            action_type=action_type,
            target=target,
            value=value,
            value2=value2,
            raw_text=match.group(0),
            is_valid=is_valid,
            error_message=error
        )

    @classmethod
    def _parse_args(cls, args_str: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """Parse argument string into (target, value, value2)."""
        if not args_str.strip():
            return None, None, None

        args = [a.strip() for a in args_str.split(",")]

        target = None
        value = None
        value2 = None

        for i, arg in enumerate(args):
            if i == 0:
                # First arg is usually target (constraint/variable name)
                # Try to parse as float first
                try:
                    value = float(arg)
                except ValueError:
                    target = arg.strip("'\"")
            elif i == 1:
                try:
                    value = float(arg)
                except ValueError:
                    pass
            elif i == 2:
                try:
                    value2 = float(arg)
                except ValueError:
                    pass

        return target, value, value2

    @classmethod
    def _validate_args(
        cls,
        action_type: str,
        target: Optional[str],
        value: Optional[float],
        value2: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """Validate arguments for action type."""

        # Actions requiring target
        target_required = {"RELAX_CONSTRAINT", "DROP_CONSTRAINT", "UPDATE_RHS", "UPDATE_BOUNDS", "CHECK_SLACK"}

        # Actions requiring value
        value_required = {"RELAX_CONSTRAINT", "UPDATE_RHS"}

        # Actions requiring two values
        two_values_required = {"UPDATE_BOUNDS"}

        if action_type in target_required and not target:
            return False, f"{action_type} requires a target constraint/variable"

        if action_type in value_required and value is None:
            return False, f"{action_type} requires a value"

        if action_type in two_values_required and (value is None or value2 is None):
            return False, f"{action_type} requires two values (lb, ub)"

        return True, None

    @classmethod
    def extract_diagnosis(cls, completion: str) -> List[str]:
        """
        Extract diagnosed constraints from <think> tags.

        Args:
            completion: LLM completion with <think> reasoning

        Returns:
            List of constraint names mentioned in diagnosis
        """
        match = cls.THINK_PATTERN.search(completion)
        if not match:
            return []

        think_content = match.group(1)

        # Extract constraint names (patterns like 'c1', 'c_key', etc.)
        constraint_pattern = re.compile(r"['\"]?(c\w*|constraint_\w+)['\"]?", re.IGNORECASE)
        constraints = constraint_pattern.findall(think_content)

        return list(set(constraints))

    @classmethod
    def extract_think_content(cls, completion: str) -> Optional[str]:
        """Extract content from <think> tags."""
        match = cls.THINK_PATTERN.search(completion)
        return match.group(1).strip() if match else None


def extract_diagnosis(completion: str) -> List[str]:
    """Convenience function for extracting diagnosis from completion."""
    return ActionParser.extract_diagnosis(completion)
