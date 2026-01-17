"""
Composite reward function with explicit diagnosis component.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-17_phase4_composite_reward.md

Key Components:
    - Outcome reward (50%): OPTIMAL=+100, INFEASIBLE=-50
    - Diagnosis reward (30%): Explicit DA signal based on IIS identification
    - Efficiency reward (20%): Step-count based reward

This provides explicit gradient signal for diagnosis accuracy improvement,
addressing Phase 3.2 finding that GRPO cannot improve DA when reward
variance is low.

Example:
    >>> from src.training.composite_reward import composite_reward_func
    >>> trainer = GRPOTrainer(reward_funcs=composite_reward_func, ...)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.action_parser import ActionParser, extract_diagnosis

logger = logging.getLogger(__name__)

# Reward weights (sum to 1.0)
WEIGHT_OUTCOME = 0.5
WEIGHT_DIAGNOSIS = 0.3
WEIGHT_EFFICIENCY = 0.2

# Reward values
REWARD_CONFIG = {
    # Outcome rewards
    "success_reward": 100.0,
    "failure_reward": -50.0,
    "syntax_error_reward": -50.0,

    # Diagnosis rewards (scaled 0-20 based on DA score)
    "max_diagnosis_reward": 20.0,

    # Efficiency rewards
    "fast_solve_reward": 10.0,      # ≤2 steps
    "moderate_solve_reward": 5.0,   # 3-5 steps
    "slow_solve_penalty": -2.0,     # per step beyond 5
}


def compute_diagnosis_score(
    completion: str,
    ground_truth_iis: List[str],
) -> float:
    """
    Compute diagnosis accuracy score.

    Args:
        completion: Model's completion text
        ground_truth_iis: Ground truth IIS constraints

    Returns:
        float: DA score in [0, 1]
    """
    if not ground_truth_iis:
        return 0.5  # No ground truth, give neutral score

    # Extract diagnosed constraints from completion
    diagnosed = extract_diagnosis(completion)

    if not diagnosed:
        return 0.0  # No diagnosis provided

    # Convert to lowercase for comparison
    gt_lower = {c.lower() for c in ground_truth_iis}
    diagnosed_lower = {c.lower() for c in diagnosed}

    # Compute intersection over union (Jaccard similarity)
    intersection = len(gt_lower & diagnosed_lower)
    union = len(gt_lower | diagnosed_lower)

    if union == 0:
        return 0.0

    return intersection / union


def compute_outcome_reward(
    parsed,
    model_file: Optional[str] = None,
) -> float:
    """
    Compute outcome reward using Gurobi solver verification.

    Args:
        parsed: Parsed action
        model_file: Path to MIP model file

    Returns:
        float: Outcome reward (+100 for success, -50 for failure)
    """
    if not parsed.is_valid:
        return REWARD_CONFIG["syntax_error_reward"]

    action_type = parsed.action_type

    # Only verify repair actions
    if action_type not in ["RELAX_CONSTRAINT", "DROP_CONSTRAINT"]:
        # Return heuristic reward for non-repair actions
        if action_type == "SUBMIT":
            return 5.0
        elif action_type == "GET_IIS":
            return 8.0
        else:
            return 0.0

    # Try solver verification if model file available
    if model_file and os.path.exists(model_file):
        try:
            import gurobipy as gp

            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0)
                env.setParam("LogToConsole", 0)
                env.start()
                m = gp.read(model_file, env)

            # Apply action
            if action_type == "RELAX_CONSTRAINT":
                constr = m.getConstrByName(parsed.target)
                if constr:
                    delta = parsed.delta if parsed.delta else 1.0
                    constr.RHS += delta
            elif action_type == "DROP_CONSTRAINT":
                constr = m.getConstrByName(parsed.target)
                if constr:
                    m.remove(constr)

            m.update()
            m.setParam("TimeLimit", 10)
            m.optimize()

            if m.status == gp.GRB.OPTIMAL:
                return REWARD_CONFIG["success_reward"]
            elif m.status == gp.GRB.INFEASIBLE:
                return REWARD_CONFIG["failure_reward"]
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Solver verification failed: {e}")
            # Fall through to heuristic

    # Heuristic reward based on action quality
    import random
    base_reward = 20.0 if parsed.target else 5.0
    return base_reward + random.uniform(-2, 2)


def compute_efficiency_reward(steps: int) -> float:
    """
    Compute efficiency reward based on number of steps.

    Args:
        steps: Number of steps taken

    Returns:
        float: Efficiency reward
    """
    if steps <= 2:
        return REWARD_CONFIG["fast_solve_reward"]
    elif steps <= 5:
        return REWARD_CONFIG["moderate_solve_reward"]
    else:
        excess = steps - 5
        return REWARD_CONFIG["slow_solve_penalty"] * excess


def compute_composite_reward(
    completion: str,
    ground_truth_iis: List[str],
    model_file: Optional[str] = None,
    steps: int = 1,
) -> dict:
    """
    Compute weighted composite reward.

    Args:
        completion: Model's completion text
        ground_truth_iis: Ground truth IIS constraints
        model_file: Path to MIP model file
        steps: Number of steps taken

    Returns:
        dict: Breakdown of reward components and total
    """
    # Parse action
    parsed = ActionParser.parse(completion)

    # 1. Outcome reward (50% weight)
    r_outcome = compute_outcome_reward(parsed, model_file)

    # 2. Diagnosis reward (30% weight)
    da_score = compute_diagnosis_score(completion, ground_truth_iis)
    r_diagnosis = REWARD_CONFIG["max_diagnosis_reward"] * da_score

    # 3. Efficiency reward (20% weight)
    r_efficiency = compute_efficiency_reward(steps)

    # Weighted sum
    total = (
        WEIGHT_OUTCOME * r_outcome +
        WEIGHT_DIAGNOSIS * r_diagnosis +
        WEIGHT_EFFICIENCY * r_efficiency
    )

    return {
        "outcome": r_outcome,
        "diagnosis": r_diagnosis,
        "efficiency": r_efficiency,
        "da_score": da_score,
        "total": total,
        "weights": {
            "outcome": WEIGHT_OUTCOME,
            "diagnosis": WEIGHT_DIAGNOSIS,
            "efficiency": WEIGHT_EFFICIENCY,
        },
    }


def composite_reward_func(
    prompts,
    completions: List[str],
    problem_id: Optional[List[str]] = None,
    model_file: Optional[List[str]] = None,
    iis_constraints: Optional[List] = None,
    step_counts: Optional[List[int]] = None,
    **kwargs
) -> List[float]:
    """
    Composite reward function for TRL GRPOTrainer.

    This function computes a weighted combination of:
    1. Outcome reward (50%): Based on solver verification
    2. Diagnosis reward (30%): Based on IIS identification accuracy
    3. Efficiency reward (20%): Based on step count

    Args:
        prompts: List of input prompts
        completions: List of generated completions
        problem_id: List of problem IDs
        model_file: List of MIP model file paths
        iis_constraints: List of ground truth IIS constraints
        step_counts: List of step counts for each completion
        **kwargs: Additional dataset columns

    Returns:
        List[float]: Composite rewards for each completion
    """
    rewards = []

    for i, completion in enumerate(completions):
        try:
            # Handle completion format
            if isinstance(completion, list):
                completion_str = str(completion)
            else:
                completion_str = str(completion)

            # Parse IIS constraints
            gt_iis = []
            if iis_constraints is not None and i < len(iis_constraints):
                iis_val = iis_constraints[i]
                if isinstance(iis_val, str):
                    try:
                        gt_iis = json.loads(iis_val)
                    except:
                        gt_iis = []
                elif isinstance(iis_val, list):
                    gt_iis = iis_val

            # Get model file
            mf = model_file[i] if model_file and i < len(model_file) else None

            # Get step count (default to 1)
            steps = step_counts[i] if step_counts and i < len(step_counts) else 1

            # Compute composite reward
            reward_breakdown = compute_composite_reward(
                completion=completion_str,
                ground_truth_iis=gt_iis,
                model_file=mf,
                steps=steps,
            )

            rewards.append(reward_breakdown["total"])

        except Exception as e:
            logger.warning(f"Error computing composite reward for sample {i}: {e}")
            rewards.append(REWARD_CONFIG["syntax_error_reward"])

    return rewards


# Separate reward functions for multi-reward training
def outcome_reward_only(
    prompts,
    completions: List[str],
    model_file: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """Outcome-only reward function (for comparison)."""
    rewards = []
    for i, completion in enumerate(completions):
        parsed = ActionParser.parse(str(completion))
        mf = model_file[i] if model_file and i < len(model_file) else None
        reward = compute_outcome_reward(parsed, mf)
        rewards.append(reward)
    return rewards


def diagnosis_reward_only(
    prompts,
    completions: List[str],
    iis_constraints: Optional[List] = None,
    **kwargs
) -> List[float]:
    """Diagnosis-only reward function (for comparison)."""
    rewards = []
    for i, completion in enumerate(completions):
        gt_iis = []
        if iis_constraints is not None and i < len(iis_constraints):
            iis_val = iis_constraints[i]
            if isinstance(iis_val, str):
                try:
                    gt_iis = json.loads(iis_val)
                except:
                    gt_iis = []
            elif isinstance(iis_val, list):
                gt_iis = iis_val

        da_score = compute_diagnosis_score(str(completion), gt_iis)
        reward = REWARD_CONFIG["max_diagnosis_reward"] * da_score
        rewards.append(reward)
    return rewards


def efficiency_reward_only(
    prompts,
    completions: List[str],
    step_counts: Optional[List[int]] = None,
    **kwargs
) -> List[float]:
    """Efficiency-only reward function (for comparison)."""
    rewards = []
    for i, _ in enumerate(completions):
        steps = step_counts[i] if step_counts and i < len(step_counts) else 1
        reward = compute_efficiency_reward(steps)
        rewards.append(reward)
    return rewards


# Multi-reward configuration for TRL
def get_multi_reward_funcs():
    """
    Get list of reward functions for TRL multi-reward training.

    Returns:
        List of (name, weight, reward_func) tuples
    """
    return [
        ("outcome", WEIGHT_OUTCOME, outcome_reward_only),
        ("diagnosis", WEIGHT_DIAGNOSIS, diagnosis_reward_only),
        ("efficiency", WEIGHT_EFFICIENCY, efficiency_reward_only),
    ]
