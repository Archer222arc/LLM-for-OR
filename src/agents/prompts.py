"""
Prompt templates for LLM-based debugging agents.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - SYSTEM_PROMPT: System prompt defining the agent's role
    - format_state: Convert DebugState to readable format
    - format_history: Format action history

Example:
    >>> from src.agents.prompts import SYSTEM_PROMPT, format_state
    >>> prompt = format_state(state)
"""

from typing import List, Dict, Any

from src.environments import DebugState


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert Operations Research debugger. Your task is to diagnose and repair infeasible optimization models.

## Your Goal
Transform an INFEASIBLE model into an OPTIMAL one by identifying and fixing constraint conflicts.

## Available Actions

### Diagnosis Actions (gather information)
- GET_IIS: Compute the Irreducible Infeasible Subsystem - a minimal set of conflicting constraints
- CHECK_SLACK(constraint): Check the slack value for a specific constraint

### Repair Actions (modify the model)
- DROP_CONSTRAINT(constraint): Remove a constraint from the model
- RELAX_CONSTRAINT(constraint, epsilon): Relax constraint bounds by epsilon
- UPDATE_RHS(constraint, value): Update the right-hand side value of a constraint
- UPDATE_BOUNDS(variable, lb, ub): Update variable bounds

### Meta Actions
- RESET: Reset the model to its original state
- SUBMIT: Submit the current model as your solution

## Strategy
1. First, use GET_IIS to identify the conflicting constraints
2. Analyze which constraint is most likely incorrect
3. Apply a targeted repair action
4. Check if the model is now feasible
5. Repeat until OPTIMAL, then SUBMIT

## Response Format
Always respond with a valid JSON object:
```json
{
    "reasoning": "Brief explanation of your decision (1-2 sentences)",
    "action": "ACTION_NAME",
    "target": "constraint_or_variable_name",
    "value": 0.0,
    "value2": 0.0
}
```

Notes:
- "target" is required for CHECK_SLACK, DROP_CONSTRAINT, RELAX_CONSTRAINT, UPDATE_RHS, UPDATE_BOUNDS
- "value" is required for RELAX_CONSTRAINT (epsilon), UPDATE_RHS (new_rhs), UPDATE_BOUNDS (lb)
- "value2" is required for UPDATE_BOUNDS (ub)
- For GET_IIS, RESET, SUBMIT: only "action" and "reasoning" are needed
"""


# =============================================================================
# State Formatting
# =============================================================================

def format_state(state: DebugState) -> str:
    """
    Format DebugState for LLM consumption.

    Args:
        state: Current environment state

    Returns:
        Formatted string representation
    """
    lines = [
        "## Current State",
        f"- Solver Status: {state.solver_status}",
        f"- Step: {state.step_count}",
    ]

    # IIS information
    if state.iis_constraints or state.iis_bounds:
        lines.append("")
        lines.append("## IIS (Irreducible Infeasible Subsystem)")
        if state.iis_constraints:
            lines.append(f"- Conflicting Constraints: {state.iis_constraints}")
        if state.iis_bounds:
            lines.append(f"- Conflicting Bounds: {state.iis_bounds}")
    else:
        if state.solver_status == "INFEASIBLE":
            lines.append("")
            lines.append("## IIS")
            lines.append("- Not yet computed. Use GET_IIS to identify conflicts.")

    # Model information
    lines.append("")
    lines.append("## Model Structure")
    lines.append(f"- Total Constraints: {len(state.constraint_names)}")
    lines.append(f"- Total Variables: {len(state.variable_names)}")

    if state.constraint_names and len(state.constraint_names) <= 10:
        lines.append(f"- Constraint Names: {state.constraint_names}")
    elif state.constraint_names:
        lines.append(f"- Constraint Names (first 10): {state.constraint_names[:10]}")

    # History
    if state.history:
        lines.append("")
        lines.append("## Recent Actions")
        lines.append(format_history(state.history[-5:]))  # Last 5 actions

    # Objective (if available)
    if state.objective is not None:
        lines.append("")
        lines.append(f"## Objective Value: {state.objective}")

    lines.append("")
    lines.append("What action should be taken next?")

    return "\n".join(lines)


def format_history(history: List[Dict[str, Any]]) -> str:
    """
    Format action history for LLM consumption.

    Args:
        history: List of action-result pairs

    Returns:
        Formatted string representation
    """
    if not history:
        return "No previous actions."

    lines = []
    for entry in history:
        step = entry.get("step", "?")
        action = entry.get("action", {})
        result = entry.get("result", {})

        action_type = action.get("action_type", "unknown")
        target = action.get("target", "")
        value = action.get("value", "")

        # Format action string
        if target and value:
            action_str = f"{action_type}({target}, {value})"
        elif target:
            action_str = f"{action_type}({target})"
        else:
            action_str = f"{action_type}()"

        # Format result
        success = result.get("success", False)
        result_str = "success" if success else "failed"
        if "error" in result:
            result_str = f"error: {result['error']}"

        lines.append(f"  Step {step}: {action_str} -> {result_str}")

    return "\n".join(lines)


# =============================================================================
# Response Parsing Helpers
# =============================================================================

VALID_ACTIONS = {
    "GET_IIS", "get_iis",
    "CHECK_SLACK", "check_slack",
    "DROP_CONSTRAINT", "drop_constraint",
    "RELAX_CONSTRAINT", "relax_constraint",
    "UPDATE_RHS", "update_rhs",
    "UPDATE_BOUNDS", "update_bounds",
    "RESET", "reset",
    "SUBMIT", "submit",
}


def normalize_action_name(action: str) -> str:
    """
    Normalize action name to lowercase format.

    Args:
        action: Action name from LLM response

    Returns:
        Normalized action name
    """
    return action.lower().strip()


def validate_action_response(response: Dict[str, Any]) -> bool:
    """
    Validate that LLM response contains required fields.

    Args:
        response: Parsed JSON response

    Returns:
        True if valid, False otherwise
    """
    if "action" not in response:
        return False

    action = response["action"]
    if action not in VALID_ACTIONS:
        return False

    normalized = normalize_action_name(action)

    # Check required fields based on action type
    needs_target = normalized in {
        "check_slack", "drop_constraint", "relax_constraint",
        "update_rhs", "update_bounds"
    }
    needs_value = normalized in {
        "relax_constraint", "update_rhs", "update_bounds"
    }

    if needs_target and not response.get("target"):
        return False

    if needs_value and response.get("value") is None:
        return False

    return True


# =============================================================================
# Few-Shot Examples
# =============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "type": "Transportation LP",
        "description": "Transportation problem where total demand exceeds total supply",
        "initial_status": "INFEASIBLE",
        "iis": ["demand_A", "demand_B", "supply_total"],
        "diagnosis": "The demand constraints require more goods than available supply",
        "steps": [
            {
                "action": "GET_IIS",
                "result": "IIS: demand_A, demand_B, supply_total"
            },
            {
                "action": "RELAX_CONSTRAINT",
                "target": "demand_A",
                "value": 5.0,
                "reasoning": "Reduce demand at location A to match available supply",
                "result": "OPTIMAL"
            }
        ],
        "final_status": "OPTIMAL"
    },
    {
        "type": "Assignment MIP",
        "description": "Assignment problem where capacity is insufficient for required assignments",
        "initial_status": "INFEASIBLE",
        "iis": ["assign_worker_1", "assign_worker_2", "capacity_machine"],
        "diagnosis": "Machine capacity constraint conflicts with worker assignment requirements",
        "steps": [
            {
                "action": "GET_IIS",
                "result": "IIS: assign_worker_1, assign_worker_2, capacity_machine"
            },
            {
                "action": "DROP_CONSTRAINT",
                "target": "capacity_machine",
                "reasoning": "Remove overly restrictive capacity constraint",
                "result": "OPTIMAL"
            }
        ],
        "final_status": "OPTIMAL"
    },
    {
        "type": "Production Planning LP",
        "description": "Production plan with conflicting resource and demand constraints",
        "initial_status": "INFEASIBLE",
        "iis": ["resource_limit", "min_demand"],
        "diagnosis": "Minimum demand requirement exceeds available resource capacity",
        "steps": [
            {
                "action": "GET_IIS",
                "result": "IIS: resource_limit, min_demand"
            },
            {
                "action": "UPDATE_RHS",
                "target": "resource_limit",
                "value": 150.0,
                "reasoning": "Increase resource availability to meet minimum demand",
                "result": "OPTIMAL"
            }
        ],
        "final_status": "OPTIMAL"
    }
]


def format_few_shot_prompt(n_examples: int = 2) -> str:
    """
    Format few-shot examples for inclusion in the prompt.

    Args:
        n_examples: Number of examples to include (1-3)

    Returns:
        Formatted string with examples
    """
    n_examples = min(n_examples, len(FEW_SHOT_EXAMPLES))
    examples = FEW_SHOT_EXAMPLES[:n_examples]

    lines = [
        "",
        "## Examples",
        ""
    ]

    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i}: {ex['type']}")
        lines.append(f"**Problem**: {ex['description']}")
        lines.append(f"**Initial Status**: {ex['initial_status']}")
        lines.append(f"**IIS**: {ex['iis']}")
        lines.append(f"**Diagnosis**: {ex['diagnosis']}")
        lines.append("")
        lines.append("**Solution Steps**:")

        for step in ex['steps']:
            action = step['action']
            target = step.get('target', '')
            value = step.get('value', '')
            result = step.get('result', '')
            reasoning = step.get('reasoning', '')

            if target and value:
                lines.append(f"1. {action}({target}, {value})")
            elif target:
                lines.append(f"1. {action}({target})")
            else:
                lines.append(f"1. {action}()")

            if reasoning:
                lines.append(f"   Reasoning: {reasoning}")
            lines.append(f"   Result: {result}")

        lines.append(f"**Final Status**: {ex['final_status']}")
        lines.append("")

    return "\n".join(lines)


def get_system_prompt(use_few_shot: bool = False, n_examples: int = 2) -> str:
    """
    Get the system prompt, optionally with few-shot examples.

    Args:
        use_few_shot: Whether to include few-shot examples
        n_examples: Number of examples if use_few_shot is True

    Returns:
        Complete system prompt string
    """
    if use_few_shot:
        return SYSTEM_PROMPT + format_few_shot_prompt(n_examples)
    return SYSTEM_PROMPT
