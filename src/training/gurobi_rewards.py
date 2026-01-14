"""
Gurobi solver-backed reward functions for TRL GRPO training.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/05_TRAINING.md

Key Components:
    - gurobi_reward_func: Main reward function for TRL GRPOTrainer
    - Composite reward: outcome + process + faithfulness
    - Full Gurobi verification for accurate outcome reward

Example:
    >>> from src.training.gurobi_rewards import gurobi_reward_func
    >>> trainer = GRPOTrainer(reward_funcs=gurobi_reward_func, ...)

    # Enable full solver verification (slower but accurate)
    >>> set_use_solver_verification(True)
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.action_parser import ActionParser, ParsedAction, extract_diagnosis

logger = logging.getLogger(__name__)

# Global flag for solver verification mode
_USE_SOLVER_VERIFICATION = False


def set_use_solver_verification(enabled: bool):
    """Enable/disable full Gurobi solver verification for outcome reward."""
    global _USE_SOLVER_VERIFICATION
    _USE_SOLVER_VERIFICATION = enabled
    logger.info(f"Solver verification {'enabled' if enabled else 'disabled'}")


def get_use_solver_verification() -> bool:
    """Check if solver verification is enabled."""
    return _USE_SOLVER_VERIFICATION


# Reward configuration (matches RewardConfig in environments/reward.py)
REWARD_CONFIG = {
    # Outcome rewards (sparse)
    "success_reward": 100.0,      # OPTIMAL achieved
    "failure_reward": -50.0,      # Still INFEASIBLE
    "syntax_error_reward": -50.0, # Invalid action

    # Process rewards (dense)
    "iis_reduction_reward": 10.0, # Per IIS constraint removed
    "step_penalty": -1.0,         # Per-step cost

    # Faithfulness penalties
    "diagnosis_contradiction_penalty": -20.0,  # Wrong diagnosis
}


def gurobi_reward_func(
    prompts,  # Can be List[str] or List[List[Dict]] for chat format
    completions: List[str],
    problem_id: Optional[List[str]] = None,
    model_file: Optional[List[str]] = None,
    iis_constraints: Optional[List] = None,  # Can be List[str] (JSON) or List[List]
    **kwargs
) -> List[float]:
    """
    Main reward function for TRL GRPOTrainer using Gurobi solver.

    This function computes rewards by:
    1. Parsing actions from completions
    2. Applying actions to MIP models via Gurobi
    3. Computing composite reward (outcome + process + faithfulness)

    Args:
        prompts: List of input prompts (str or chat messages)
        completions: List of generated completions
        problem_id: List of problem IDs (from dataset)
        model_file: List of MIP model file paths (from dataset)
        iis_constraints: List of current IIS constraints (from dataset)
        **kwargs: Additional dataset columns

    Returns:
        List[float]: Rewards for each completion
    """
    rewards = []

    for i, completion in enumerate(completions):
        try:
            # Handle completion that might be a list of tokens or string
            if isinstance(completion, list):
                # If it's a list, try to join or convert
                completion_str = str(completion)
            else:
                completion_str = str(completion)

            # Handle iis_constraints that might already be a list
            iis_val = None
            if iis_constraints is not None:
                iis_val = iis_constraints[i] if i < len(iis_constraints) else None
                # Convert to string if it's already a list
                if isinstance(iis_val, list):
                    import json
                    iis_val = json.dumps(iis_val)

            reward = _compute_single_reward(
                completion=completion_str,
                problem_id=problem_id[i] if problem_id and i < len(problem_id) else None,
                model_file=model_file[i] if model_file and i < len(model_file) else None,
                iis_constraints=iis_val,
            )
            rewards.append(reward)
        except Exception as e:
            logger.warning(f"Error computing reward for sample {i}: {e}")
            rewards.append(REWARD_CONFIG["syntax_error_reward"])

    return rewards


def _compute_single_reward(
    completion: str,
    problem_id: Optional[str],
    model_file: Optional[str],
    iis_constraints: Optional[str],
) -> float:
    """
    Compute reward for a single completion.

    Args:
        completion: Generated completion text
        problem_id: Problem identifier
        model_file: Path to MIP model file
        iis_constraints: Current IIS constraints (JSON string)

    Returns:
        float: Composite reward
    """
    # 1. Parse action from completion
    parsed = ActionParser.parse(completion)

    if not parsed.is_valid:
        logger.debug(f"Invalid action: {parsed.error_message}")
        return REWARD_CONFIG["syntax_error_reward"]

    # 2. Compute outcome reward based on action type
    outcome_reward = _compute_outcome_reward(parsed, model_file)

    # 3. Compute process reward
    process_reward = _compute_process_reward(parsed, iis_constraints)

    # 4. Compute faithfulness penalty
    faithfulness_penalty = _compute_faithfulness_penalty(
        completion, iis_constraints
    )

    total = outcome_reward + process_reward + faithfulness_penalty

    logger.debug(
        f"Reward breakdown - outcome: {outcome_reward:.1f}, "
        f"process: {process_reward:.1f}, faithfulness: {faithfulness_penalty:.1f}, "
        f"total: {total:.1f}"
    )

    return total


def _compute_outcome_reward(
    parsed: ParsedAction,
    model_file: Optional[str]
) -> float:
    """
    Compute outcome reward using Gurobi solver verification.

    Two modes:
    1. Full verification (USE_SOLVER_VERIFICATION=True):
       - Load MIP model, apply action, re-solve
       - Return +100 for OPTIMAL, -50 for INFEASIBLE
    2. Heuristic mode (USE_SOLVER_VERIFICATION=False):
       - Quick estimate based on action type
       - Used for fast iteration during development
    """
    if _USE_SOLVER_VERIFICATION and model_file:
        return _compute_outcome_reward_with_solver(parsed, model_file)
    else:
        return _compute_outcome_reward_heuristic(parsed)


def _compute_outcome_reward_with_solver(
    parsed: ParsedAction,
    model_file: str
) -> float:
    """
    Full Gurobi verification for accurate outcome reward.

    This is the core implementation for Novelty 3 (Solver as RLVR Oracle).
    """
    try:
        import gurobipy as gp
    except ImportError:
        logger.warning("Gurobi not available, falling back to heuristic")
        return _compute_outcome_reward_heuristic(parsed)

    if not os.path.exists(model_file):
        logger.warning(f"Model file not found: {model_file}")
        return _compute_outcome_reward_heuristic(parsed)

    action_type = parsed.action_type

    # Only verify repair actions (RELAX_CONSTRAINT, DROP_CONSTRAINT)
    if action_type not in ["RELAX_CONSTRAINT", "DROP_CONSTRAINT"]:
        return _compute_outcome_reward_heuristic(parsed)

    try:
        # 1. Load model with suppressed output
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("LogToConsole", 0)
            env.start()
            m = gp.read(model_file, env)

        # 2. Apply action
        if action_type == "RELAX_CONSTRAINT":
            constr = m.getConstrByName(parsed.target)
            if constr is None:
                logger.debug(f"Constraint not found: {parsed.target}")
                return REWARD_CONFIG["failure_reward"]
            # Relax by delta (default 1.0 if not specified)
            delta = parsed.delta if parsed.delta else 1.0
            constr.RHS += delta

        elif action_type == "DROP_CONSTRAINT":
            constr = m.getConstrByName(parsed.target)
            if constr is None:
                logger.debug(f"Constraint not found: {parsed.target}")
                return REWARD_CONFIG["failure_reward"]
            m.remove(constr)

        # 3. Re-solve
        m.update()
        m.setParam("TimeLimit", 10)  # 10 second timeout
        m.optimize()

        # 4. Check status and return appropriate reward
        if m.status == gp.GRB.OPTIMAL:
            logger.debug(f"Action led to OPTIMAL: {parsed.action_type}({parsed.target})")
            return REWARD_CONFIG["success_reward"]  # +100
        elif m.status == gp.GRB.INFEASIBLE:
            logger.debug(f"Action still INFEASIBLE: {parsed.action_type}({parsed.target})")
            return REWARD_CONFIG["failure_reward"]  # -50
        elif m.status == gp.GRB.TIME_LIMIT:
            logger.debug("Solver timeout, assuming progress made")
            return 10.0  # Partial credit
        else:
            logger.debug(f"Unexpected status: {m.status}")
            return 0.0

    except gp.GurobiError as e:
        logger.warning(f"Gurobi error: {e}")
        return REWARD_CONFIG["syntax_error_reward"]
    except Exception as e:
        logger.warning(f"Error in solver verification: {e}")
        return _compute_outcome_reward_heuristic(parsed)


def _compute_outcome_reward_heuristic(parsed: ParsedAction) -> float:
    """
    Quick heuristic estimate for outcome reward (no solver execution).

    Used when:
    - Solver verification is disabled
    - Model file is not available
    - For fast development iteration
    """
    action_type = parsed.action_type

    if action_type == "SUBMIT":
        return 0.0  # Neutral - actual outcome unknown

    elif action_type in ["RELAX_CONSTRAINT", "DROP_CONSTRAINT"]:
        if parsed.target:
            return 20.0  # Partial credit for valid repair action
        return 5.0

    elif action_type == "GET_IIS":
        return 5.0  # Small positive for gathering info

    elif action_type in ["CHECK_SLACK", "RESET"]:
        return 0.0

    return 0.0


def _compute_process_reward(
    parsed: ParsedAction,
    iis_constraints: Optional[str]
) -> float:
    """
    Compute process reward for dense credit assignment.
    """
    reward = REWARD_CONFIG["step_penalty"]  # Base step penalty

    # Parse IIS constraints if provided
    current_iis = []
    if iis_constraints:
        try:
            import json
            current_iis = json.loads(iis_constraints) if isinstance(iis_constraints, str) else iis_constraints
        except:
            pass

    # Bonus for targeting IIS constraints
    if parsed.action_type in ["RELAX_CONSTRAINT", "DROP_CONSTRAINT"]:
        if parsed.target and parsed.target in current_iis:
            # Targeting an actual IIS constraint - good!
            reward += REWARD_CONFIG["iis_reduction_reward"]
        elif parsed.target:
            # Targeting non-IIS constraint - suboptimal but not wrong
            reward += 2.0

    return reward


def _compute_faithfulness_penalty(
    completion: str,
    iis_constraints: Optional[str]
) -> float:
    """
    Compute faithfulness penalty for diagnosis consistency.

    Penalizes when the model's stated diagnosis contradicts actual IIS.
    """
    # Extract diagnosed constraints from <think> tags
    diagnosed = extract_diagnosis(completion)

    if not diagnosed:
        return 0.0  # No diagnosis to check

    # Parse actual IIS
    actual_iis = []
    if iis_constraints:
        try:
            import json
            actual_iis = json.loads(iis_constraints) if isinstance(iis_constraints, str) else iis_constraints
        except:
            pass

    if not actual_iis:
        return 0.0  # No ground truth to compare

    # Check if diagnosed constraints are actually in IIS
    for constraint in diagnosed:
        if constraint.lower() not in [c.lower() for c in actual_iis]:
            # Diagnosed a constraint not in IIS - hallucination
            return REWARD_CONFIG["diagnosis_contradiction_penalty"]

    return 0.0  # Faithful diagnosis


# Multi-reward function variants for TRL
def outcome_reward(
    completions: List[str],
    problem_id: Optional[List[str]] = None,
    model_file: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """Outcome-only reward function."""
    rewards = []
    for i, completion in enumerate(completions):
        parsed = ActionParser.parse(completion)
        reward = _compute_outcome_reward(
            parsed,
            model_file[i] if model_file else None
        )
        rewards.append(reward)
    return rewards


def process_reward(
    completions: List[str],
    iis_constraints: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """Process-only reward function."""
    rewards = []
    for i, completion in enumerate(completions):
        parsed = ActionParser.parse(completion)
        reward = _compute_process_reward(
            parsed,
            iis_constraints[i] if iis_constraints else None
        )
        rewards.append(reward)
    return rewards


def faithfulness_reward(
    completions: List[str],
    iis_constraints: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """Faithfulness-only reward function."""
    rewards = []
    for i, completion in enumerate(completions):
        reward = _compute_faithfulness_penalty(
            completion,
            iis_constraints[i] if iis_constraints else None
        )
        rewards.append(reward)
    return rewards
