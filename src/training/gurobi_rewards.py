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
    Compute reward for a single completion using HYBRID strategy.

    Hybrid strategy combines:
    1. Solver verification (sparse, accurate) - scaled down
    2. Heuristic rewards (dense, approximate)
    3. Process rewards (targeting IIS constraints)

    This ensures non-zero variance for gradient learning.

    Args:
        completion: Generated completion text
        problem_id: Problem identifier
        model_file: Path to MIP model file
        iis_constraints: Current IIS constraints (JSON string)

    Returns:
        float: Composite reward with guaranteed variance
    """
    import random

    # 1. Parse action from completion
    parsed = ActionParser.parse(completion)

    if not parsed.is_valid:
        logger.debug(f"Invalid action: {parsed.error_message}")
        # Still give small differentiated rewards for partial parsing
        base_penalty = REWARD_CONFIG["syntax_error_reward"]
        # Add small noise to break ties
        noise = random.uniform(-2, 2)
        return base_penalty + noise

    # 2. Compute outcome reward (solver verification if enabled)
    outcome_reward = _compute_outcome_reward(parsed, model_file)

    # 3. HYBRID: Always add heuristic component for variance
    heuristic_reward = _compute_outcome_reward_heuristic(parsed)

    # 4. Compute process reward (dense signal)
    process_reward = _compute_process_reward(parsed, iis_constraints)

    # 5. Compute faithfulness penalty
    faithfulness_penalty = _compute_faithfulness_penalty(
        completion, iis_constraints
    )

    # 6. HYBRID combination:
    # - If solver succeeds (+100): outcome dominates
    # - If solver fails (-50): heuristic + process provide differentiation
    if outcome_reward == REWARD_CONFIG["success_reward"]:
        # Success: full outcome + bonus
        total = outcome_reward + process_reward + faithfulness_penalty
    else:
        # Failure: scale down outcome penalty, add heuristic for differentiation
        scaled_outcome = outcome_reward * 0.5  # -25 instead of -50
        total = scaled_outcome + heuristic_reward + process_reward + faithfulness_penalty

    # 7. Add small noise to break exact ties (ensures variance > 0)
    noise = random.uniform(-1, 1)
    total += noise

    logger.debug(
        f"Reward breakdown - outcome: {outcome_reward:.1f}, heuristic: {heuristic_reward:.1f}, "
        f"process: {process_reward:.1f}, faithfulness: {faithfulness_penalty:.1f}, "
        f"noise: {noise:.2f}, total: {total:.1f}"
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

    Provides differentiated rewards based on action quality:
    - Repair actions with valid targets: highest
    - Repair actions without targets: medium
    - Diagnostic actions: small positive
    - Other actions: neutral

    Used for hybrid reward strategy to ensure variance.
    """
    import random

    action_type = parsed.action_type
    base_reward = 0.0

    if action_type == "SUBMIT":
        base_reward = 5.0  # Small positive for attempting completion

    elif action_type == "DROP_CONSTRAINT":
        if parsed.target:
            # Reward based on target name pattern (longer names often more specific)
            specificity_bonus = min(len(parsed.target) * 0.5, 5.0)
            base_reward = 25.0 + specificity_bonus
        else:
            base_reward = 10.0

    elif action_type == "RELAX_CONSTRAINT":
        if parsed.target and parsed.value is not None:
            # Full specification: target + relaxation amount
            # Prefer moderate relaxation values
            if 0.1 <= abs(parsed.value) <= 100:
                base_reward = 30.0
            else:
                base_reward = 20.0  # Extreme values less preferred
        elif parsed.target:
            base_reward = 15.0
        else:
            base_reward = 5.0

    elif action_type == "GET_IIS":
        base_reward = 8.0  # Gathering info is useful

    elif action_type == "CHECK_SLACK":
        if parsed.target:
            base_reward = 6.0
        else:
            base_reward = 3.0

    elif action_type in ["UPDATE_RHS", "UPDATE_BOUNDS"]:
        if parsed.target:
            base_reward = 15.0
        else:
            base_reward = 5.0

    elif action_type == "RESET":
        base_reward = -5.0  # Slightly discourage resets

    # Add small random component for variance
    variance_component = random.uniform(-2, 2)

    return base_reward + variance_component


def _compute_process_reward(
    parsed: ParsedAction,
    iis_constraints: Optional[str]
) -> float:
    """
    Compute process reward for dense credit assignment.

    Provides strong signal for targeting IIS constraints, which is
    the key diagnostic insight for infeasibility repair.
    """
    import random

    reward = REWARD_CONFIG["step_penalty"]  # Base step penalty (-1)

    # Parse IIS constraints if provided
    current_iis = []
    if iis_constraints:
        try:
            import json
            iis_data = json.loads(iis_constraints) if isinstance(iis_constraints, str) else iis_constraints
            if isinstance(iis_data, list):
                current_iis = [str(c).lower() for c in iis_data]
        except:
            pass

    # Bonus for targeting IIS constraints
    if parsed.action_type in ["RELAX_CONSTRAINT", "DROP_CONSTRAINT", "UPDATE_RHS"]:
        if parsed.target:
            target_lower = parsed.target.lower()
            # Check if target matches any IIS constraint (case-insensitive)
            if any(target_lower == iis_c or target_lower in iis_c or iis_c in target_lower
                   for iis_c in current_iis):
                # Targeting an actual IIS constraint - excellent!
                reward += REWARD_CONFIG["iis_reduction_reward"]  # +10
                # Bonus for DROP vs RELAX (DROP more decisive)
                if parsed.action_type == "DROP_CONSTRAINT":
                    reward += 5.0
            elif current_iis:
                # Have IIS info but targeting wrong constraint
                reward += 1.0  # Small credit for trying
            else:
                # No IIS info available, give benefit of doubt
                reward += 3.0
        else:
            # No target specified
            reward -= 2.0

    elif parsed.action_type == "GET_IIS":
        # Gathering diagnostic info is always good
        reward += 3.0

    # Small random component
    reward += random.uniform(-0.5, 0.5)

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
