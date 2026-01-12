"""
Unit tests for evaluation module.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A4_Evaluation_Metrics.md

Tests:
    - EpisodeResult data class
    - MetricsCalculator methods
    - EpisodeTracker functionality
    - BenchmarkRunner integration
"""

import pytest
import gurobipy as gp

from src.solvers import GurobiSolver
from src.environments import SolverDebugEnv, Action, ActionType, DebugState
from src.agents import HeuristicAgent, RandomAgent
from src.evaluation import (
    EpisodeResult,
    BenchmarkConfig,
    MetricsCalculator,
    EpisodeTracker,
    BenchmarkProblem,
    BenchmarkRunner,
    aggregate_trajectories,
    extract_action_sequence,
    compute_action_diversity,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_results():
    """Create sample episode results for testing."""
    return [
        EpisodeResult(
            success=True,
            final_status="OPTIMAL",
            steps=3,
            total_reward=107.0,
            trajectory=[
                {"step": 1, "action_type": "get_iis", "reward": -1.0},
                {"step": 2, "action_type": "drop_constraint", "reward": 8.0},
                {"step": 3, "action_type": "submit", "reward": 100.0},
            ],
            iis_actions=["c1"],
            agent_name="HeuristicAgent",
            problem_id="problem_1",
        ),
        EpisodeResult(
            success=True,
            final_status="OPTIMAL",
            steps=2,
            total_reward=118.0,
            trajectory=[
                {"step": 1, "action_type": "drop_constraint", "reward": 18.0},
                {"step": 2, "action_type": "submit", "reward": 100.0},
            ],
            iis_actions=["c2"],
            agent_name="HeuristicAgent",
            problem_id="problem_2",
        ),
        EpisodeResult(
            success=False,
            final_status="INFEASIBLE",
            steps=5,
            total_reward=-55.0,
            trajectory=[
                {"step": 1, "action_type": "get_iis", "reward": -1.0},
                {"step": 2, "action_type": "drop_constraint", "reward": -1.0},
                {"step": 3, "action_type": "drop_constraint", "reward": -1.0},
                {"step": 4, "action_type": "drop_constraint", "reward": -2.0},
                {"step": 5, "action_type": "submit", "reward": -50.0},
            ],
            iis_actions=["c3", "c4"],
            agent_name="HeuristicAgent",
            problem_id="problem_3",
        ),
    ]


@pytest.fixture
def simple_infeasible_model():
    """Create a simple infeasible model."""
    m = gp.Model("simple_infeasible")
    m.Params.OutputFlag = 0

    x = m.addVar(lb=0, ub=10, name="x")

    # Contradicting constraints
    m.addConstr(x >= 8, name="x_lower")
    m.addConstr(x <= 5, name="x_upper")

    m.setObjective(x, gp.GRB.MAXIMIZE)
    m.update()

    return m


# =============================================================================
# EpisodeResult Tests
# =============================================================================


class TestEpisodeResult:
    """Tests for EpisodeResult data class."""

    def test_creation(self):
        """Test basic creation."""
        result = EpisodeResult(
            success=True,
            final_status="OPTIMAL",
            steps=3,
            total_reward=100.0,
        )
        assert result.success is True
        assert result.final_status == "OPTIMAL"
        assert result.steps == 3
        assert result.total_reward == 100.0

    def test_default_values(self):
        """Test default values."""
        result = EpisodeResult(
            success=False,
            final_status="INFEASIBLE",
            steps=1,
            total_reward=-50.0,
        )
        assert result.trajectory == []
        assert result.iis_actions == []
        assert result.ground_truth_fix is None
        assert result.agent_name == ""
        assert result.problem_id == ""

    def test_to_dict(self):
        """Test serialization to dict."""
        result = EpisodeResult(
            success=True,
            final_status="OPTIMAL",
            steps=2,
            total_reward=110.0,
            trajectory=[{"step": 1}, {"step": 2}],
            agent_name="TestAgent",
            problem_id="test_1",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["final_status"] == "OPTIMAL"
        assert d["trajectory_length"] == 2
        assert d["agent_name"] == "TestAgent"


# =============================================================================
# MetricsCalculator Tests
# =============================================================================


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_recovery_rate(self, sample_results):
        """Test recovery rate calculation."""
        calc = MetricsCalculator()
        rate = calc.compute_recovery_rate(sample_results)
        assert rate == pytest.approx(2 / 3)  # 2 successes out of 3

    def test_recovery_rate_empty(self):
        """Test recovery rate with empty results."""
        calc = MetricsCalculator()
        assert calc.compute_recovery_rate([]) == 0.0

    def test_avg_steps(self, sample_results):
        """Test average steps calculation."""
        calc = MetricsCalculator()
        avg = calc.compute_avg_steps(sample_results)
        assert avg == pytest.approx((3 + 2 + 5) / 3)

    def test_median_steps(self, sample_results):
        """Test median steps calculation."""
        calc = MetricsCalculator()
        median = calc.compute_median_steps(sample_results)
        assert median == 3.0  # Median of [2, 3, 5]

    def test_avg_reward(self, sample_results):
        """Test average reward calculation."""
        calc = MetricsCalculator()
        avg = calc.compute_avg_reward(sample_results)
        expected = (107.0 + 118.0 + (-55.0)) / 3
        assert avg == pytest.approx(expected)

    def test_step_efficiency(self, sample_results):
        """Test step efficiency calculation."""
        calc = MetricsCalculator()
        efficiency = calc.compute_step_efficiency(sample_results)
        avg_reward = (107.0 + 118.0 + (-55.0)) / 3
        avg_steps = (3 + 2 + 5) / 3
        assert efficiency == pytest.approx(avg_reward / avg_steps)

    def test_success_steps(self, sample_results):
        """Test success steps calculation."""
        calc = MetricsCalculator()
        success_steps = calc.compute_success_steps(sample_results)
        assert success_steps == pytest.approx((3 + 2) / 2)  # Only successful ones

    def test_success_steps_no_success(self):
        """Test success steps with no successes."""
        calc = MetricsCalculator()
        results = [
            EpisodeResult(success=False, final_status="INFEASIBLE", steps=5, total_reward=-50.0)
        ]
        assert calc.compute_success_steps(results) is None

    def test_diagnosis_accuracy(self, sample_results):
        """Test diagnosis accuracy calculation."""
        calc = MetricsCalculator()
        ground_truth = {
            "problem_1": ["c1"],
            "problem_2": ["c2"],
            "problem_3": ["c3", "c5"],  # c4 is wrong
        }
        accuracy = calc.compute_diagnosis_accuracy(sample_results, ground_truth)
        # problem_1: 1/1, problem_2: 1/1, problem_3: 1/2
        # Total: 3/4 = 0.75
        assert accuracy == pytest.approx(3 / 4)

    def test_diagnosis_accuracy_no_ground_truth(self, sample_results):
        """Test diagnosis accuracy without ground truth."""
        calc = MetricsCalculator()
        assert calc.compute_diagnosis_accuracy(sample_results, None) is None

    def test_compute_summary(self, sample_results):
        """Test comprehensive summary."""
        calc = MetricsCalculator()
        summary = calc.compute_summary(sample_results)

        assert summary["n_episodes"] == 3
        assert summary["recovery_rate"] == pytest.approx(2 / 3)
        assert "avg_steps" in summary
        assert "median_steps" in summary
        assert "avg_reward" in summary
        assert "step_efficiency" in summary
        assert "success_avg_steps" in summary

    def test_format_summary(self, sample_results):
        """Test summary formatting."""
        calc = MetricsCalculator()
        summary = calc.compute_summary(sample_results)
        formatted = calc.format_summary(summary)

        assert "OR-Debug-Bench" in formatted
        assert "Recovery Rate" in formatted
        assert "Avg Steps" in formatted


# =============================================================================
# EpisodeTracker Tests
# =============================================================================


class TestEpisodeTracker:
    """Tests for EpisodeTracker."""

    def test_creation(self):
        """Test tracker creation."""
        tracker = EpisodeTracker(agent_name="TestAgent", problem_id="test_1")
        assert tracker.agent_name == "TestAgent"
        assert tracker.problem_id == "test_1"

    def test_record_step(self):
        """Test step recording."""
        tracker = EpisodeTracker()

        state = DebugState(solver_status="INFEASIBLE", iis_constraints=["c1"])
        action = Action(ActionType.DROP_CONSTRAINT, target="c1")
        next_state = DebugState(solver_status="OPTIMAL")

        tracker.record_step(state, action, 108.0, next_state, done=True)

        result = tracker.finalize(success=True)
        assert result.steps == 1
        assert result.total_reward == 108.0
        assert "c1" in result.iis_actions

    def test_reset(self):
        """Test tracker reset."""
        tracker = EpisodeTracker()

        state = DebugState(solver_status="INFEASIBLE")
        action = Action(ActionType.GET_IIS)
        tracker.record_step(state, action, -1.0, state, done=False)

        tracker.reset()

        result = tracker.finalize(success=False)
        assert result.steps == 0
        assert result.total_reward == 0.0

    def test_finalize(self):
        """Test finalization."""
        tracker = EpisodeTracker(
            agent_name="Agent1",
            problem_id="prob_1",
            ground_truth_fix="Drop c1",
        )

        state = DebugState(solver_status="INFEASIBLE")
        next_state = DebugState(solver_status="OPTIMAL")
        action = Action(ActionType.SUBMIT)

        tracker.record_step(state, action, 100.0, next_state, done=True)

        result = tracker.finalize(success=True)
        assert result.agent_name == "Agent1"
        assert result.problem_id == "prob_1"
        assert result.ground_truth_fix == "Drop c1"
        assert result.success is True


# =============================================================================
# Trajectory Aggregation Tests
# =============================================================================


class TestTrajectoryAggregation:
    """Tests for trajectory aggregation functions."""

    def test_aggregate_trajectories(self, sample_results):
        """Test trajectory aggregation."""
        agg = aggregate_trajectories(sample_results)

        assert agg["total_actions"] == 10  # 3 + 2 + 5
        assert "get_iis" in agg["action_counts"]
        assert "drop_constraint" in agg["action_counts"]
        assert "submit" in agg["action_counts"]

    def test_aggregate_empty(self):
        """Test aggregation with empty results."""
        agg = aggregate_trajectories([])
        assert agg == {}

    def test_extract_action_sequence(self, sample_results):
        """Test action sequence extraction."""
        seq = extract_action_sequence(sample_results[0])
        assert seq == ["get_iis", "drop_constraint", "submit"]

    def test_compute_action_diversity(self, sample_results):
        """Test action diversity computation."""
        diversity = compute_action_diversity(sample_results)
        # Each episode has 2-3 unique action types
        assert diversity > 0


# =============================================================================
# BenchmarkRunner Tests
# =============================================================================


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_creation(self):
        """Test runner creation."""
        runner = BenchmarkRunner()
        assert runner.config.max_steps == 50
        assert runner.results == []

    def test_creation_with_config(self):
        """Test runner creation with config."""
        config = BenchmarkConfig(max_steps=10, verbose=True)
        runner = BenchmarkRunner(config=config)
        assert runner.config.max_steps == 10
        assert runner.config.verbose is True

    def test_run_episode(self, simple_infeasible_model):
        """Test running a single episode."""
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver, max_steps=10)
        agent = HeuristicAgent()

        runner = BenchmarkRunner()
        result = runner.run_episode(env, agent, problem_id="test_problem")

        assert isinstance(result, EpisodeResult)
        assert result.problem_id == "test_problem"
        assert result.agent_name == "HeuristicAgent"
        assert result.steps > 0

    def test_run_benchmark(self, simple_infeasible_model):
        """Test running a benchmark."""
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver, max_steps=10)

        problems = [
            BenchmarkProblem(problem_id="p1", env=env),
        ]

        runner = BenchmarkRunner()
        agent = HeuristicAgent()

        results = runner.run_benchmark(problems, agent)

        assert len(results) == 1
        assert len(runner.results) == 1

    def test_get_summary(self, simple_infeasible_model):
        """Test getting summary after benchmark."""
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver, max_steps=10)

        problems = [BenchmarkProblem(problem_id="p1", env=env)]

        runner = BenchmarkRunner()
        runner.run_benchmark(problems, HeuristicAgent())

        summary = runner.get_summary()
        assert "n_episodes" in summary
        assert "recovery_rate" in summary

    def test_format_summary(self, simple_infeasible_model):
        """Test format summary."""
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver, max_steps=10)

        problems = [BenchmarkProblem(problem_id="p1", env=env)]

        runner = BenchmarkRunner()
        runner.run_benchmark(problems, HeuristicAgent())

        formatted = runner.format_summary()
        assert "OR-Debug-Bench" in formatted

    def test_clear_results(self, simple_infeasible_model):
        """Test clearing results."""
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver, max_steps=10)

        problems = [BenchmarkProblem(problem_id="p1", env=env)]

        runner = BenchmarkRunner()
        runner.run_benchmark(problems, HeuristicAgent())
        assert len(runner.results) > 0

        runner.clear_results()
        assert len(runner.results) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluationIntegration:
    """Integration tests for evaluation module."""

    def test_full_evaluation_pipeline(self, simple_infeasible_model):
        """Test full evaluation pipeline."""
        # Create environment
        solver = GurobiSolver.from_model(simple_infeasible_model)
        env = SolverDebugEnv(solver, max_steps=10)

        # Create benchmark problems
        problems = [
            BenchmarkProblem(
                problem_id="test_1",
                env=env,
                ground_truth_fix="Drop x_lower or x_upper",
            ),
        ]

        # Run benchmark
        config = BenchmarkConfig(max_steps=10, n_episodes=1)
        runner = BenchmarkRunner(config=config)
        agent = HeuristicAgent()

        results = runner.run_benchmark(problems, agent)

        # Check results
        assert len(results) == 1
        assert results[0].agent_name == "HeuristicAgent"
        assert results[0].problem_id == "test_1"

        # Get summary
        summary = runner.get_summary()
        assert summary["n_episodes"] == 1

        # Format and print
        formatted = runner.format_summary()
        assert len(formatted) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
