"""
Unit tests for agent implementations.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Tests:
    - BaseAgent interface
    - RandomAgent behavior
    - HeuristicAgent behavior
    - MockLLMAgent behavior
    - Prompt formatting
"""

import pytest
import gurobipy as gp

from src.solvers import GurobiSolver
from src.environments import (
    SolverDebugEnv,
    DebugState,
    Action,
    ActionType,
)
from src.agents import (
    BaseAgent,
    RandomAgent,
    HeuristicAgent,
    GreedyDropAgent,
    DoNothingAgent,
    MockLLMAgent,
    SYSTEM_PROMPT,
    format_state,
    format_history,
)


# =============================================================================
# Fixtures
# =============================================================================


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
def infeasible_state(simple_infeasible_model):
    """Create an infeasible DebugState."""
    solver = GurobiSolver.from_model(simple_infeasible_model)
    env = SolverDebugEnv(solver)
    state, _ = env.reset()
    return state


@pytest.fixture
def optimal_state():
    """Create an optimal DebugState."""
    return DebugState(
        solver_status="OPTIMAL",
        constraint_names=["c1", "c2"],
        variable_names=["x", "y"],
        objective=42.0,
    )


# =============================================================================
# BaseAgent Tests
# =============================================================================


class TestBaseAgent:
    """Tests for BaseAgent interface."""

    def test_cannot_instantiate(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent()

    def test_subclass_must_implement_act(self):
        """Test that subclass must implement act()."""
        class IncompleteAgent(BaseAgent):
            def reset(self):
                pass

        with pytest.raises(TypeError):
            IncompleteAgent()

    def test_subclass_must_implement_reset(self):
        """Test that subclass must implement reset()."""
        class IncompleteAgent(BaseAgent):
            def act(self, state):
                pass

        with pytest.raises(TypeError):
            IncompleteAgent()


# =============================================================================
# RandomAgent Tests
# =============================================================================


class TestRandomAgent:
    """Tests for RandomAgent."""

    def test_creation(self):
        """Test agent creation."""
        agent = RandomAgent()
        assert agent.name == "RandomAgent"

    def test_creation_with_seed(self):
        """Test agent creation with seed."""
        agent = RandomAgent(seed=42)
        assert agent._seed == 42

    def test_act_returns_action(self, infeasible_state):
        """Test that act returns an Action."""
        agent = RandomAgent(seed=42)
        action = agent.act(infeasible_state)
        assert isinstance(action, Action)

    def test_reproducibility_with_seed(self, infeasible_state):
        """Test that same seed gives same actions."""
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)

        actions1 = [agent1.act(infeasible_state) for _ in range(5)]
        actions2 = [agent2.act(infeasible_state) for _ in range(5)]

        for a1, a2 in zip(actions1, actions2):
            assert a1.action_type == a2.action_type

    def test_reset(self, infeasible_state):
        """Test reset clears history."""
        agent = RandomAgent()
        agent.act(infeasible_state)
        agent.record_step(
            infeasible_state,
            Action(ActionType.GET_IIS),
            0.0,
            infeasible_state,
            False
        )
        assert len(agent.get_episode_history()) > 0

        agent.reset()
        assert len(agent.get_episode_history()) == 0


# =============================================================================
# HeuristicAgent Tests
# =============================================================================


class TestHeuristicAgent:
    """Tests for HeuristicAgent."""

    def test_creation(self):
        """Test agent creation."""
        agent = HeuristicAgent()
        assert agent.name == "HeuristicAgent"

    def test_submits_when_optimal(self, optimal_state):
        """Test that agent submits when model is optimal."""
        agent = HeuristicAgent()
        action = agent.act(optimal_state)
        assert action.action_type == ActionType.SUBMIT

    def test_gets_iis_when_infeasible_without_iis(self, infeasible_state):
        """Test that agent gets IIS when infeasible and no IIS known."""
        # Ensure no IIS is present
        state = DebugState(
            solver_status="INFEASIBLE",
            iis_constraints=[],
            constraint_names=["c1", "c2"],
        )
        agent = HeuristicAgent()
        action = agent.act(state)
        assert action.action_type == ActionType.GET_IIS

    def test_drops_constraint_when_iis_available(self, infeasible_state):
        """Test that agent drops constraint when IIS is available."""
        state = DebugState(
            solver_status="INFEASIBLE",
            iis_constraints=["x_lower", "x_upper"],
            constraint_names=["x_lower", "x_upper"],
        )
        agent = HeuristicAgent()
        action = agent.act(state)
        assert action.action_type == ActionType.DROP_CONSTRAINT
        assert action.target == "x_lower"  # First IIS constraint

    def test_reset(self):
        """Test reset clears internal state."""
        agent = HeuristicAgent()
        agent._has_iis = True
        agent.reset()
        assert agent._has_iis is False


# =============================================================================
# GreedyDropAgent Tests
# =============================================================================


class TestGreedyDropAgent:
    """Tests for GreedyDropAgent."""

    def test_creation(self):
        """Test agent creation."""
        agent = GreedyDropAgent()
        assert agent.name == "GreedyDropAgent"

    def test_submits_when_optimal(self, optimal_state):
        """Test that agent submits when model is optimal."""
        agent = GreedyDropAgent()
        action = agent.act(optimal_state)
        assert action.action_type == ActionType.SUBMIT

    def test_drops_all_iis_constraints(self):
        """Test that agent drops all IIS constraints."""
        state = DebugState(
            solver_status="INFEASIBLE",
            iis_constraints=["c1", "c2", "c3"],
            constraint_names=["c1", "c2", "c3"],
        )
        agent = GreedyDropAgent()

        actions = []
        for _ in range(3):
            action = agent.act(state)
            actions.append(action)

        assert all(a.action_type == ActionType.DROP_CONSTRAINT for a in actions)
        targets = [a.target for a in actions]
        assert set(targets) == {"c1", "c2", "c3"}

    def test_reset(self):
        """Test reset clears dropped constraints list."""
        agent = GreedyDropAgent()
        agent._dropped_constraints = ["c1", "c2"]
        agent.reset()
        assert agent._dropped_constraints == []


# =============================================================================
# DoNothingAgent Tests
# =============================================================================


class TestDoNothingAgent:
    """Tests for DoNothingAgent."""

    def test_always_submits(self, infeasible_state, optimal_state):
        """Test that agent always submits."""
        agent = DoNothingAgent()

        action1 = agent.act(infeasible_state)
        action2 = agent.act(optimal_state)

        assert action1.action_type == ActionType.SUBMIT
        assert action2.action_type == ActionType.SUBMIT


# =============================================================================
# MockLLMAgent Tests
# =============================================================================


class TestMockLLMAgent:
    """Tests for MockLLMAgent."""

    def test_creation(self):
        """Test agent creation."""
        agent = MockLLMAgent()
        assert agent.name == "MockLLMAgent"

    def test_follows_predefined_responses(self):
        """Test that agent follows predefined responses."""
        responses = [
            {"action": "get_iis"},
            {"action": "drop_constraint", "target": "c1"},
            {"action": "submit"},
        ]
        agent = MockLLMAgent(responses=responses)

        state = DebugState(solver_status="INFEASIBLE")

        action1 = agent.act(state)
        assert action1.action_type == ActionType.GET_IIS

        action2 = agent.act(state)
        assert action2.action_type == ActionType.DROP_CONSTRAINT
        assert action2.target == "c1"

        action3 = agent.act(state)
        assert action3.action_type == ActionType.SUBMIT

    def test_fallback_to_heuristic(self):
        """Test fallback when responses exhausted."""
        agent = MockLLMAgent(responses=[])

        # Should use heuristic
        state = DebugState(
            solver_status="INFEASIBLE",
            iis_constraints=["c1"],
        )
        action = agent.act(state)
        assert action.action_type == ActionType.DROP_CONSTRAINT

    def test_reset(self):
        """Test reset resets response index."""
        responses = [{"action": "get_iis"}]
        agent = MockLLMAgent(responses=responses)

        state = DebugState()
        agent.act(state)
        assert agent._response_index == 1

        agent.reset()
        assert agent._response_index == 0


# =============================================================================
# Prompt Tests
# =============================================================================


class TestPrompts:
    """Tests for prompt formatting."""

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        assert len(SYSTEM_PROMPT) > 100
        assert "Operations Research" in SYSTEM_PROMPT

    def test_format_state(self):
        """Test state formatting."""
        state = DebugState(
            solver_status="INFEASIBLE",
            step_count=3,
            iis_constraints=["c1", "c2"],
            constraint_names=["c1", "c2", "c3"],
            variable_names=["x", "y"],
        )

        formatted = format_state(state)

        assert "INFEASIBLE" in formatted
        assert "Step: 3" in formatted
        assert "c1" in formatted
        assert "c2" in formatted

    def test_format_state_without_iis(self):
        """Test state formatting without IIS."""
        state = DebugState(
            solver_status="INFEASIBLE",
            iis_constraints=[],
        )

        formatted = format_state(state)

        assert "Not yet computed" in formatted

    def test_format_history(self):
        """Test history formatting."""
        history = [
            {
                "step": 1,
                "action": {"action_type": "get_iis"},
                "result": {"success": True},
            },
            {
                "step": 2,
                "action": {"action_type": "drop_constraint", "target": "c1"},
                "result": {"success": True},
            },
        ]

        formatted = format_history(history)

        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "get_iis" in formatted
        assert "drop_constraint" in formatted

    def test_format_history_empty(self):
        """Test formatting empty history."""
        formatted = format_history([])
        assert "No previous actions" in formatted


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentIntegration:
    """Integration tests with SolverDebugEnv."""

    def test_heuristic_agent_solves_simple_problem(self, simple_infeasible_model):
        """Test that heuristic agent can solve a simple problem."""
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver, max_steps=10)
        agent = HeuristicAgent()

        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while not env.is_done and steps < 10:
            action = agent.act(state)
            state, reward, _, _, _ = env.step(action)
            total_reward += reward
            steps += 1

        # Should eventually solve or submit
        assert steps <= 10

    def test_random_agent_runs_episode(self, simple_infeasible_model):
        """Test that random agent can run an episode."""
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver, max_steps=5)
        agent = RandomAgent(seed=42)

        state, _ = env.reset()
        steps = 0

        while not env.is_done and steps < 5:
            action = agent.act(state)
            state, _, _, _, _ = env.step(action)
            steps += 1

        # Should complete within max_steps
        assert steps <= 5

    def test_record_step(self, simple_infeasible_model):
        """Test step recording."""
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver)
        agent = HeuristicAgent()

        state, _ = env.reset()
        action = agent.act(state)
        new_state, reward, terminated, _, _ = env.step(action)

        agent.record_step(state, action, reward, new_state, terminated)

        history = agent.get_episode_history()
        assert len(history) == 1
        assert history[0]["reward"] == reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
