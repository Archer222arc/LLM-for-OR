"""
Gymnasium-style MDP environment for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - SolverDebugEnv: Main environment class with reset/step interface

Example:
    >>> from src.solvers import GurobiSolver
    >>> from src.environments import SolverDebugEnv, Action, ActionType
    >>> solver = GurobiSolver.from_file("model.mps")
    >>> env = SolverDebugEnv(solver, problem_nl="Minimize cost...")
    >>> state, info = env.reset()
    >>> action = Action(ActionType.GET_IIS)
    >>> state, reward, done, truncated, info = env.step(action)
"""

from typing import Tuple, Optional, Dict, Any, List
import random

from src.solvers import GurobiSolver, SolverState, IISResult

from .state import DebugState, StepResult
from .action import Action, ActionType
from .reward import RewardCalculator, RewardConfig


class SolverDebugEnv:
    """
    OR-Debug-Bench MDP environment using Gymnasium-style interface.

    This environment wraps a GurobiSolver and provides an MDP interface
    for debugging infeasible optimization models.

    State Space:
        - problem_nl: Natural language problem description
        - solver_status: OPTIMAL/INFEASIBLE/UNBOUNDED/etc.
        - iis_constraints: Constraints in the IIS
        - iis_bounds: Variable bounds in the IIS
        - history: Previous action-result pairs

    Action Space:
        - Diagnosis: GET_IIS, CHECK_SLACK
        - Repair: RELAX_CONSTRAINT, DROP_CONSTRAINT, UPDATE_RHS, UPDATE_BOUNDS
        - Meta: RESET, SUBMIT

    Rewards:
        - Outcome: +100 (OPTIMAL), -50 (still INFEASIBLE)
        - Process: +10 (IIS reduced), -1 (step penalty)
        - Faithfulness: -20 (hallucination penalty)
    """

    def __init__(
        self,
        solver: GurobiSolver,
        problem_nl: str = "",
        max_steps: int = 50,
        reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize SolverDebugEnv.

        Args:
            solver: GurobiSolver instance (may be feasible or infeasible)
            problem_nl: Natural language problem description
            max_steps: Maximum steps per episode
            reward_config: Configuration for reward calculation
            seed: Random seed for reproducibility
        """
        self._solver = solver
        self._problem_nl = problem_nl
        self._max_steps = max_steps
        self._reward_calculator = RewardCalculator(reward_config)

        # State tracking
        self._state: Optional[DebugState] = None
        self._step_count = 0
        self._done = False

        # Random seed
        if seed is not None:
            random.seed(seed)

    # =========================================================================
    # Gymnasium Interface
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[DebugState, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed
            options: Additional reset options

        Returns:
            Tuple of (initial_state, info_dict)
        """
        if seed is not None:
            random.seed(seed)

        # Reset solver to original state
        self._solver.reset()

        # Solve to get current status
        solver_state = self._solver.solve()

        # Compute IIS if infeasible
        iis_constraints = []
        iis_bounds = []
        if solver_state.status == "INFEASIBLE":
            iis_result = self._solver.compute_iis()
            iis_constraints = iis_result.constraints
            iis_bounds = iis_result.bounds

        # Build initial state
        self._state = DebugState(
            problem_nl=self._problem_nl,
            solver_status=solver_state.status,
            iis_constraints=iis_constraints,
            iis_bounds=iis_bounds,
            constraint_names=self._solver.get_all_constraints(),
            variable_names=self._solver.get_all_variables(),
            history=[],
            step_count=0,
            objective=solver_state.objective,
            gap=solver_state.gap,
        )

        self._step_count = 0
        self._done = False

        info = {
            "solver_status": solver_state.status,
            "iis_size": len(iis_constraints) + len(iis_bounds),
            "solve_time": solver_state.solve_time,
        }

        return self._state, info

    def step(self, action: Action) -> Tuple[DebugState, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (new_state, reward, terminated, truncated, info)
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._done:
            raise RuntimeError("Episode has ended. Call reset() to start new episode.")

        old_state = self._state.copy()
        self._step_count += 1

        # Execute action
        action_result = self._execute_action(action)

        # Get new solver state
        solver_state = self._solver.solve()

        # Compute IIS if infeasible
        iis_constraints = []
        iis_bounds = []
        if solver_state.status == "INFEASIBLE":
            iis_result = self._solver.compute_iis()
            iis_constraints = iis_result.constraints
            iis_bounds = iis_result.bounds

        # Build new state
        new_state = DebugState(
            problem_nl=self._problem_nl,
            solver_status=solver_state.status,
            iis_constraints=iis_constraints,
            iis_bounds=iis_bounds,
            constraint_names=self._solver.get_all_constraints(),
            variable_names=self._solver.get_all_variables(),
            history=old_state.history.copy(),
            step_count=self._step_count,
            objective=solver_state.objective,
            gap=solver_state.gap,
        )

        # Add action to history
        new_state.add_to_history(action.to_dict(), action_result)

        # Check termination conditions
        terminated = self._check_terminated(new_state, action)
        truncated = self._step_count >= self._max_steps

        self._done = terminated or truncated

        # Compute reward
        reward = self._reward_calculator.compute_reward(
            old_state, new_state, action, is_terminal=self._done
        )

        self._state = new_state

        info = {
            "action_result": action_result,
            "solver_status": solver_state.status,
            "iis_size": len(iis_constraints) + len(iis_bounds),
            "solve_time": solver_state.solve_time,
            "reward_breakdown": self._reward_calculator.get_reward_breakdown(
                old_state, new_state, action, self._done
            ),
        }

        return new_state, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the environment state.

        Args:
            mode: Rendering mode ("human" for console output)

        Returns:
            String representation if mode is not "human"
        """
        if self._state is None:
            return "Environment not initialized"

        output = str(self._state)

        if mode == "human":
            print(output)
            return None
        else:
            return output

    def close(self) -> None:
        """Clean up environment resources."""
        pass  # No cleanup needed currently

    # =========================================================================
    # Action Execution
    # =========================================================================

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """
        Execute an action on the solver.

        Args:
            action: Action to execute

        Returns:
            Result dictionary with action-specific information
        """
        if action.action_type == ActionType.GET_IIS:
            return self._execute_get_iis()
        elif action.action_type == ActionType.CHECK_SLACK:
            return self._execute_check_slack(action.target)
        elif action.action_type == ActionType.RELAX_CONSTRAINT:
            return self._execute_relax_constraint(action.target, action.value)
        elif action.action_type == ActionType.DROP_CONSTRAINT:
            return self._execute_drop_constraint(action.target)
        elif action.action_type == ActionType.UPDATE_RHS:
            return self._execute_update_rhs(action.target, action.value)
        elif action.action_type == ActionType.UPDATE_BOUNDS:
            return self._execute_update_bounds(action.target, action.value, action.value2)
        elif action.action_type == ActionType.RESET:
            return self._execute_reset()
        elif action.action_type == ActionType.SUBMIT:
            return self._execute_submit()
        else:
            return {"success": False, "error": f"Unknown action: {action.action_type}"}

    def _execute_get_iis(self) -> Dict[str, Any]:
        """Execute GET_IIS action."""
        solver_state = self._solver.solve()

        if solver_state.status != "INFEASIBLE":
            return {
                "success": False,
                "error": "Cannot compute IIS: model is not infeasible",
                "status": solver_state.status,
            }

        iis_result = self._solver.compute_iis()
        return {
            "success": True,
            "constraints": iis_result.constraints,
            "bounds": iis_result.bounds,
            "size": iis_result.size,
        }

    def _execute_check_slack(self, constraint_name: str) -> Dict[str, Any]:
        """Execute CHECK_SLACK action."""
        try:
            info = self._solver.get_constraint_info(constraint_name)
            return {
                "success": True,
                "constraint": constraint_name,
                "sense": info.sense,
                "rhs": info.rhs,
                "slack": info.slack,
                "is_in_iis": info.is_in_iis,
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    def _execute_relax_constraint(
        self, constraint_name: str, epsilon: float
    ) -> Dict[str, Any]:
        """Execute RELAX_CONSTRAINT action."""
        try:
            self._solver.relax_constraint(constraint_name, epsilon)
            return {
                "success": True,
                "constraint": constraint_name,
                "epsilon": epsilon,
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    def _execute_drop_constraint(self, constraint_name: str) -> Dict[str, Any]:
        """Execute DROP_CONSTRAINT action."""
        try:
            self._solver.drop_constraint(constraint_name)
            return {
                "success": True,
                "constraint": constraint_name,
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    def _execute_update_rhs(
        self, constraint_name: str, new_rhs: float
    ) -> Dict[str, Any]:
        """Execute UPDATE_RHS action."""
        try:
            self._solver.update_rhs(constraint_name, new_rhs)
            return {
                "success": True,
                "constraint": constraint_name,
                "new_rhs": new_rhs,
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    def _execute_update_bounds(
        self, variable_name: str, lb: float, ub: float
    ) -> Dict[str, Any]:
        """Execute UPDATE_BOUNDS action."""
        try:
            self._solver.update_variable_bounds(variable_name, lb, ub)
            return {
                "success": True,
                "variable": variable_name,
                "lb": lb,
                "ub": ub,
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    def _execute_reset(self) -> Dict[str, Any]:
        """Execute RESET action."""
        self._solver.reset()
        return {"success": True}

    def _execute_submit(self) -> Dict[str, Any]:
        """Execute SUBMIT action."""
        solver_state = self._solver.solve()
        return {
            "success": solver_state.status == "OPTIMAL",
            "status": solver_state.status,
            "objective": solver_state.objective,
        }

    # =========================================================================
    # Termination Logic
    # =========================================================================

    def _check_terminated(self, state: DebugState, action: Action) -> bool:
        """
        Check if the episode should terminate.

        Args:
            state: Current state
            action: Action just taken

        Returns:
            True if episode should terminate
        """
        # Success: Model became optimal
        if state.is_optimal():
            return True

        # Explicit submission
        if action.action_type == ActionType.SUBMIT:
            return True

        return False

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> Optional[DebugState]:
        """Get current state."""
        return self._state

    @property
    def solver(self) -> GurobiSolver:
        """Get underlying solver."""
        return self._solver

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count

    @property
    def max_steps(self) -> int:
        """Get maximum steps per episode."""
        return self._max_steps

    @property
    def is_done(self) -> bool:
        """Check if episode has ended."""
        return self._done

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_valid_actions(self) -> List[Action]:
        """
        Get list of currently valid actions.

        Returns:
            List of valid Action objects
        """
        actions = []

        # Diagnosis actions are always valid
        actions.append(Action(ActionType.GET_IIS))

        # Constraint-based actions
        if self._state:
            for name in self._state.constraint_names:
                actions.append(Action(ActionType.CHECK_SLACK, target=name))
                actions.append(Action(ActionType.DROP_CONSTRAINT, target=name))
                # RELAX and UPDATE_RHS need value - omit from this list

            for name in self._state.variable_names:
                pass  # UPDATE_BOUNDS needs values - omit

        # Meta actions
        actions.append(Action(ActionType.RESET))
        actions.append(Action(ActionType.SUBMIT))

        return actions
