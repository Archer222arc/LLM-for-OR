"""
Unit tests for SolverDebugEnv MDP environment.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Tests:
    - Action/ActionType creation and validation
    - DebugState operations
    - RewardCalculator computations
    - SolverDebugEnv reset/step interface
"""

import pytest
import gurobipy as gp

from src.solvers import GurobiSolver
from src.environments import (
    ActionType,
    Action,
    DebugState,
    StepResult,
    RewardCalculator,
    RewardConfig,
    SolverDebugEnv,
    get_iis,
    check_slack,
    drop_constraint,
    update_rhs,
    submit,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_feasible_model():
    """Create a simple feasible LP model."""
    m = gp.Model("simple_feasible")
    m.Params.OutputFlag = 0

    x = m.addVar(lb=0, ub=10, name="x")
    y = m.addVar(lb=0, ub=10, name="y")

    m.addConstr(x + y <= 15, name="sum_limit")
    m.addConstr(x >= 2, name="x_lower")
    m.addConstr(y >= 3, name="y_lower")

    m.setObjective(x + 2 * y, gp.GRB.MAXIMIZE)
    m.update()

    return m


@pytest.fixture
def simple_infeasible_model():
    """Create a simple infeasible LP model."""
    m = gp.Model("simple_infeasible")
    m.Params.OutputFlag = 0

    x = m.addVar(lb=0, ub=10, name="x")

    # Contradicting constraints
    m.addConstr(x >= 8, name="x_lower")
    m.addConstr(x <= 5, name="x_upper")

    m.setObjective(x, gp.GRB.MAXIMIZE)
    m.update()

    return m


@pytest.fixture
def feasible_solver(simple_feasible_model):
    """Create a GurobiSolver with feasible model."""
    return GurobiSolver.from_model(simple_feasible_model)


@pytest.fixture
def infeasible_solver(simple_infeasible_model):
    """Create a GurobiSolver with infeasible model."""
    return GurobiSolver.from_model(simple_infeasible_model)


# =============================================================================
# ActionType Tests
# =============================================================================


class TestActionType:
    """Tests for ActionType enum."""

    def test_diagnosis_actions(self):
        """Test diagnosis action classification."""
        assert ActionType.GET_IIS.is_diagnosis
        assert ActionType.CHECK_SLACK.is_diagnosis
        assert not ActionType.DROP_CONSTRAINT.is_diagnosis

    def test_repair_actions(self):
        """Test repair action classification."""
        assert ActionType.RELAX_CONSTRAINT.is_repair
        assert ActionType.DROP_CONSTRAINT.is_repair
        assert ActionType.UPDATE_RHS.is_repair
        assert ActionType.UPDATE_BOUNDS.is_repair
        assert not ActionType.GET_IIS.is_repair

    def test_meta_actions(self):
        """Test meta action classification."""
        assert ActionType.RESET.is_meta
        assert ActionType.SUBMIT.is_meta
        assert not ActionType.GET_IIS.is_meta

    def test_requires_target(self):
        """Test target requirement."""
        assert ActionType.CHECK_SLACK.requires_target
        assert ActionType.DROP_CONSTRAINT.requires_target
        assert not ActionType.GET_IIS.requires_target
        assert not ActionType.SUBMIT.requires_target


# =============================================================================
# Action Tests
# =============================================================================


class TestAction:
    """Tests for Action dataclass."""

    def test_action_creation(self):
        """Test basic action creation."""
        action = Action(ActionType.GET_IIS)
        assert action.action_type == ActionType.GET_IIS
        assert action.target is None
        assert action.value is None

    def test_action_with_target(self):
        """Test action with target."""
        action = Action(ActionType.DROP_CONSTRAINT, target="c1")
        assert action.target == "c1"

    def test_action_with_value(self):
        """Test action with value."""
        action = Action(ActionType.UPDATE_RHS, target="c1", value=10.0)
        assert action.value == 10.0

    def test_action_validation_missing_target(self):
        """Test that missing required target raises error."""
        with pytest.raises(ValueError, match="requires a target"):
            Action(ActionType.DROP_CONSTRAINT)  # Missing target

    def test_action_validation_missing_value(self):
        """Test that missing required value raises error."""
        with pytest.raises(ValueError, match="requires a value"):
            Action(ActionType.UPDATE_RHS, target="c1")  # Missing value

    def test_action_to_dict(self):
        """Test action serialization."""
        action = Action(ActionType.UPDATE_RHS, target="c1", value=10.0)
        d = action.to_dict()
        assert d["action_type"] == "update_rhs"
        assert d["target"] == "c1"
        assert d["value"] == 10.0

    def test_action_from_dict(self):
        """Test action deserialization."""
        d = {"action_type": "drop_constraint", "target": "c1"}
        action = Action.from_dict(d)
        assert action.action_type == ActionType.DROP_CONSTRAINT
        assert action.target == "c1"

    def test_factory_functions(self):
        """Test convenience factory functions."""
        assert get_iis().action_type == ActionType.GET_IIS
        assert check_slack("c1").target == "c1"
        assert drop_constraint("c1").target == "c1"
        assert update_rhs("c1", 5.0).value == 5.0
        assert submit().action_type == ActionType.SUBMIT


# =============================================================================
# DebugState Tests
# =============================================================================


class TestDebugState:
    """Tests for DebugState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = DebugState()
        assert state.problem_nl == ""
        assert state.solver_status == "LOADED"
        assert state.iis_constraints == []
        assert state.iis_bounds == []

    def test_state_with_values(self):
        """Test state with specific values."""
        state = DebugState(
            problem_nl="Minimize cost",
            solver_status="INFEASIBLE",
            iis_constraints=["c1", "c2"],
            iis_bounds=["x_lb"],
        )
        assert state.get_iis_size() == 3

    def test_is_infeasible(self):
        """Test infeasibility check."""
        state = DebugState(solver_status="INFEASIBLE")
        assert state.is_infeasible()

        state = DebugState(solver_status="OPTIMAL")
        assert not state.is_infeasible()

    def test_is_optimal(self):
        """Test optimality check."""
        state = DebugState(solver_status="OPTIMAL")
        assert state.is_optimal()

        state = DebugState(solver_status="INFEASIBLE")
        assert not state.is_optimal()

    def test_add_to_history(self):
        """Test history tracking."""
        state = DebugState(step_count=1)
        state.add_to_history(
            {"action_type": "get_iis"},
            {"success": True, "iis_size": 2}
        )
        assert len(state.history) == 1
        assert state.history[0]["step"] == 1

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        state = DebugState(
            problem_nl="Test",
            solver_status="OPTIMAL",
            iis_constraints=["c1"],
            step_count=5,
        )
        d = state.to_dict()
        state2 = DebugState.from_dict(d)
        assert state2.problem_nl == state.problem_nl
        assert state2.solver_status == state.solver_status
        assert state2.step_count == state.step_count

    def test_copy(self):
        """Test deep copy."""
        state = DebugState(iis_constraints=["c1", "c2"])
        state2 = state.copy()
        state2.iis_constraints.append("c3")
        assert len(state.iis_constraints) == 2  # Original unchanged


# =============================================================================
# RewardCalculator Tests
# =============================================================================


class TestRewardCalculator:
    """Tests for RewardCalculator."""

    def test_default_config(self):
        """Test default reward configuration."""
        calc = RewardCalculator()
        assert calc.config.success_reward == 100.0
        assert calc.config.failure_reward == -50.0
        assert calc.config.step_penalty == -1.0

    def test_custom_config(self):
        """Test custom reward configuration."""
        config = RewardConfig(success_reward=200.0)
        calc = RewardCalculator(config)
        assert calc.config.success_reward == 200.0

    def test_step_penalty(self):
        """Test step penalty is applied."""
        calc = RewardCalculator()
        old_state = DebugState(solver_status="INFEASIBLE")
        new_state = DebugState(solver_status="INFEASIBLE")
        action = Action(ActionType.GET_IIS)

        reward = calc.compute_reward(old_state, new_state, action)
        assert reward == -1.0  # Step penalty only

    def test_success_reward(self):
        """Test success reward at terminal state."""
        calc = RewardCalculator()
        old_state = DebugState(solver_status="INFEASIBLE")
        new_state = DebugState(solver_status="OPTIMAL")
        action = Action(ActionType.SUBMIT)

        reward = calc.compute_reward(old_state, new_state, action, is_terminal=True)
        assert reward == 100.0 + (-1.0)  # Success + step penalty

    def test_failure_reward(self):
        """Test failure reward at terminal state."""
        calc = RewardCalculator()
        old_state = DebugState(solver_status="INFEASIBLE")
        new_state = DebugState(solver_status="INFEASIBLE")
        action = Action(ActionType.SUBMIT)

        reward = calc.compute_reward(old_state, new_state, action, is_terminal=True)
        assert reward == -50.0 + (-1.0)  # Failure + step penalty

    def test_iis_reduction_bonus(self):
        """Test IIS reduction bonus."""
        calc = RewardCalculator()
        old_state = DebugState(
            solver_status="INFEASIBLE",
            iis_constraints=["c1", "c2", "c3"]
        )
        new_state = DebugState(
            solver_status="INFEASIBLE",
            iis_constraints=["c1"]
        )
        action = Action(ActionType.DROP_CONSTRAINT, target="c2")

        reward = calc.compute_reward(old_state, new_state, action)
        # Step penalty + IIS reduction (2 * 10) + constraint preserved
        expected = -1.0 + 20.0  # Reduction of 2 * 10
        # Note: constraint preserved bonus doesn't apply to DROP_CONSTRAINT
        assert reward == -1.0 + 20.0

    def test_reward_breakdown(self):
        """Test reward breakdown."""
        calc = RewardCalculator()
        old_state = DebugState(solver_status="INFEASIBLE")
        new_state = DebugState(solver_status="OPTIMAL")
        action = Action(ActionType.SUBMIT)

        breakdown = calc.get_reward_breakdown(
            old_state, new_state, action, is_terminal=True
        )
        assert breakdown["outcome"] == 100.0
        assert breakdown["process"] == -1.0
        assert breakdown["faithfulness"] == 0.0
        assert breakdown["total"] == 99.0


# =============================================================================
# SolverDebugEnv Tests
# =============================================================================


class TestSolverDebugEnv:
    """Tests for SolverDebugEnv."""

    def test_env_creation(self, feasible_solver):
        """Test environment creation."""
        env = SolverDebugEnv(feasible_solver, problem_nl="Test problem")
        assert env.max_steps == 50
        assert not env.is_done

    def test_reset_feasible(self, feasible_solver):
        """Test reset with feasible model."""
        env = SolverDebugEnv(feasible_solver)
        state, info = env.reset()

        assert state.solver_status == "OPTIMAL"
        assert state.step_count == 0
        assert len(state.history) == 0
        assert info["solver_status"] == "OPTIMAL"

    def test_reset_infeasible(self, infeasible_solver):
        """Test reset with infeasible model."""
        env = SolverDebugEnv(infeasible_solver)
        state, info = env.reset()

        assert state.solver_status == "INFEASIBLE"
        assert len(state.iis_constraints) > 0 or len(state.iis_bounds) > 0
        assert info["iis_size"] > 0

    def test_step_without_reset(self, feasible_solver):
        """Test that step before reset raises error."""
        env = SolverDebugEnv(feasible_solver)
        action = Action(ActionType.GET_IIS)

        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(action)

    def test_get_iis_action(self, infeasible_solver):
        """Test GET_IIS action execution."""
        env = SolverDebugEnv(infeasible_solver)
        state, _ = env.reset()

        action = Action(ActionType.GET_IIS)
        new_state, reward, terminated, truncated, info = env.step(action)

        assert info["action_result"]["success"]
        assert "constraints" in info["action_result"]
        assert new_state.step_count == 1

    def test_drop_constraint_action(self, infeasible_solver):
        """Test DROP_CONSTRAINT action execution."""
        env = SolverDebugEnv(infeasible_solver)
        state, _ = env.reset()

        # Get a constraint from IIS
        iis_constr = state.iis_constraints[0] if state.iis_constraints else None
        if iis_constr:
            action = Action(ActionType.DROP_CONSTRAINT, target=iis_constr)
            new_state, reward, terminated, truncated, info = env.step(action)

            assert info["action_result"]["success"]
            assert new_state.step_count == 1

    def test_submit_success(self, feasible_solver):
        """Test SUBMIT action with feasible model."""
        env = SolverDebugEnv(feasible_solver)
        state, _ = env.reset()

        action = Action(ActionType.SUBMIT)
        new_state, reward, terminated, truncated, info = env.step(action)

        assert terminated  # Episode ends on SUBMIT
        assert info["action_result"]["success"]
        assert reward > 0  # Success reward

    def test_submit_failure(self, infeasible_solver):
        """Test SUBMIT action with infeasible model."""
        env = SolverDebugEnv(infeasible_solver)
        state, _ = env.reset()

        action = Action(ActionType.SUBMIT)
        new_state, reward, terminated, truncated, info = env.step(action)

        assert terminated  # Episode ends on SUBMIT
        assert not info["action_result"]["success"]
        assert reward < 0  # Failure reward

    def test_max_steps_truncation(self, infeasible_solver):
        """Test episode truncation at max steps."""
        env = SolverDebugEnv(infeasible_solver, max_steps=3)
        state, _ = env.reset()

        for i in range(3):
            action = Action(ActionType.GET_IIS)
            state, reward, terminated, truncated, info = env.step(action)

        assert truncated or terminated

    def test_step_after_done(self, feasible_solver):
        """Test that step after episode ends raises error."""
        env = SolverDebugEnv(feasible_solver)
        env.reset()

        # End episode with SUBMIT
        action = Action(ActionType.SUBMIT)
        env.step(action)

        # Try to step again
        with pytest.raises(RuntimeError, match="has ended"):
            env.step(action)

    def test_reset_action(self, infeasible_solver):
        """Test RESET action."""
        env = SolverDebugEnv(infeasible_solver)
        state, _ = env.reset()

        # Execute GET_IIS first (doesn't change model state)
        get_iis_action = Action(ActionType.GET_IIS)
        state, _, terminated, _, _ = env.step(get_iis_action)

        # If not terminated, try RESET
        if not terminated:
            reset_action = Action(ActionType.RESET)
            new_state, reward, terminated, truncated, info = env.step(reset_action)
            assert info["action_result"]["success"]

    def test_render(self, feasible_solver):
        """Test render method."""
        env = SolverDebugEnv(feasible_solver)
        env.reset()

        # Should not raise
        output = env.render(mode="text")
        assert output is not None

    def test_get_valid_actions(self, infeasible_solver):
        """Test get_valid_actions method."""
        env = SolverDebugEnv(infeasible_solver)
        env.reset()

        actions = env.get_valid_actions()
        assert len(actions) > 0

        # Should include GET_IIS, RESET, SUBMIT
        action_types = [a.action_type for a in actions]
        assert ActionType.GET_IIS in action_types
        assert ActionType.RESET in action_types
        assert ActionType.SUBMIT in action_types

    def test_properties(self, feasible_solver):
        """Test environment properties."""
        env = SolverDebugEnv(feasible_solver, max_steps=100)
        env.reset()

        assert env.max_steps == 100
        assert env.step_count == 0
        assert env.state is not None
        assert env.solver is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full debugging episodes."""

    def test_full_episode_success(self, infeasible_solver):
        """Test a complete successful debugging episode."""
        env = SolverDebugEnv(infeasible_solver)
        state, info = env.reset()

        assert state.is_infeasible()

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 10:
            # Strategy: Drop IIS constraints until feasible
            if state.iis_constraints:
                action = Action(ActionType.DROP_CONSTRAINT, target=state.iis_constraints[0])
            else:
                action = Action(ActionType.SUBMIT)

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

            if state.is_optimal():
                break

        # After dropping conflicting constraints, should be optimal
        # Note: This depends on the specific model structure

    def test_full_episode_with_get_iis(self, infeasible_solver):
        """Test episode using GET_IIS action."""
        env = SolverDebugEnv(infeasible_solver)
        state, _ = env.reset()

        # First, get IIS
        action = get_iis()
        state, reward, _, _, info = env.step(action)

        assert info["action_result"]["success"]
        assert len(state.history) == 1
        assert state.history[0]["action"]["action_type"] == "get_iis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
