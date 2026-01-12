"""
Unit tests for SaboteurAgent.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A3_Data_Generation.md

Test Coverage:
    - Type A: Constraint direction flip
    - Type B: Variable type modification
    - Type C: Expression term removal
    - Type D: Contradicting constraint addition
    - Utility methods
"""

import pytest


# Skip all tests if Gurobi is not available
gurobi_available = False
try:
    import gurobipy as gp
    # Try to create a simple model to verify license
    test_model = gp.Model()
    test_model.addVar()
    test_model.optimize()
    gurobi_available = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    not gurobi_available,
    reason="Gurobi not available or license not valid"
)


def create_feasible_model():
    """Create a simple feasible model for testing."""
    import gurobipy as gp

    m = gp.Model("test_feasible")
    m.Params.OutputFlag = 0

    x = m.addVar(lb=0, ub=10, name="x")
    y = m.addVar(lb=0, ub=10, name="y")

    m.addConstr(x + y <= 15, name="upper_bound")
    m.addConstr(x >= 2, name="lower_x")
    m.addConstr(y >= 3, name="lower_y")

    m.setObjective(x + 2 * y, gp.GRB.MAXIMIZE)
    m.update()

    return m


def create_mip_model():
    """Create a simple MIP model for testing Type B."""
    import gurobipy as gp

    m = gp.Model("test_mip")
    m.Params.OutputFlag = 0

    x = m.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=10, name="x_int")
    y = m.addVar(vtype=gp.GRB.BINARY, name="y_bin")
    z = m.addVar(lb=0, ub=10, name="z_cont")

    m.addConstr(x + 5 * y + z <= 15, name="capacity")
    m.addConstr(x >= 2, name="min_x")

    m.setObjective(x + 3 * y + z, gp.GRB.MAXIMIZE)
    m.update()

    return m


class TestErrorTypes:
    """Tests for ErrorType enum and InjectionResult."""

    def test_error_type_values(self):
        """Test ErrorType enum values."""
        from src.data_generation import ErrorType

        assert ErrorType.TYPE_A.value == "A"
        assert ErrorType.TYPE_B.value == "B"
        assert ErrorType.TYPE_C.value == "C"
        assert ErrorType.TYPE_D.value == "D"

    def test_error_type_description(self):
        """Test ErrorType descriptions."""
        from src.data_generation import ErrorType

        assert "Bound Error" in ErrorType.TYPE_A.description
        assert "Variable Error" in ErrorType.TYPE_B.description
        assert "Logic Error" in ErrorType.TYPE_C.description
        assert "Conflict Error" in ErrorType.TYPE_D.description

    def test_injection_result_to_dict(self):
        """Test InjectionResult serialization."""
        from src.data_generation import ErrorType, InjectionResult

        result = InjectionResult(
            success=True,
            error_type=ErrorType.TYPE_A,
            target_name="test_constraint",
            original_value="<=",
            modified_value=">=",
            solver_status="INFEASIBLE",
            ground_truth_fix="Change back to <="
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["error_type"] == "A"
        assert d["target_name"] == "test_constraint"


class TestSaboteurAgentCreation:
    """Tests for SaboteurAgent creation."""

    def test_create_saboteur(self):
        """Test creating a SaboteurAgent."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)
        saboteur = SaboteurAgent(solver)

        assert saboteur is not None
        assert saboteur.solver is solver

    def test_create_with_seed(self):
        """Test creating with reproducible seed."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)
        saboteur = SaboteurAgent(solver, seed=42)

        assert saboteur is not None


class TestTypeAInjection:
    """Tests for Type A (constraint direction flip)."""

    def test_type_a_injection(self):
        """Test Type A injection flips constraint sense."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent, ErrorType

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)

        # Verify model is initially feasible
        state = solver.solve()
        assert state.status == "OPTIMAL"

        saboteur = SaboteurAgent(solver, seed=42)
        result = saboteur.inject_type_a()

        assert result.error_type == ErrorType.TYPE_A
        assert result.target_name in ["upper_bound", "lower_x", "lower_y"]
        assert result.original_value in ["<=", ">="]
        assert result.modified_value in ["<=", ">="]
        assert result.original_value != result.modified_value
        assert "Change" in result.ground_truth_fix

    def test_type_a_causes_infeasibility(self):
        """Test that Type A can cause infeasibility."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent
        import gurobipy as gp

        # Create a tight model where flipping will cause infeasibility
        m = gp.Model("tight")
        m.Params.OutputFlag = 0
        x = m.addVar(lb=0, name="x")
        m.addConstr(x <= 5, name="upper")
        m.addConstr(x >= 3, name="lower")
        m.setObjective(x, gp.GRB.MAXIMIZE)
        m.update()

        solver = GurobiSolver.from_model(m)
        assert solver.solve().status == "OPTIMAL"

        saboteur = SaboteurAgent(solver, seed=123)
        result = saboteur.inject_type_a()

        # After flipping, model should be infeasible
        assert result.solver_status in ["INFEASIBLE", "INF_OR_UNBD", "OPTIMAL"]


class TestTypeBInjection:
    """Tests for Type B (variable type modification)."""

    def test_type_b_injection(self):
        """Test Type B injection changes variable type."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent, ErrorType

        model = create_mip_model()
        solver = GurobiSolver.from_model(model)

        saboteur = SaboteurAgent(solver, seed=42)
        result = saboteur.inject_type_b()

        assert result.error_type == ErrorType.TYPE_B
        assert result.target_name in ["x_int", "y_bin", "z_cont"]
        assert "Change" in result.ground_truth_fix


class TestTypeCInjection:
    """Tests for Type C (expression term removal)."""

    def test_type_c_injection(self):
        """Test Type C injection removes term from constraint."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent, ErrorType

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)

        saboteur = SaboteurAgent(solver, seed=42)
        result = saboteur.inject_type_c()

        assert result.error_type == ErrorType.TYPE_C
        assert "Restore coefficient" in result.ground_truth_fix
        assert result.metadata is not None
        assert "variable" in result.metadata


class TestTypeDInjection:
    """Tests for Type D (contradicting constraint addition)."""

    def test_type_d_injection(self):
        """Test Type D injection adds contradicting constraint."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent, ErrorType

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)

        # Verify initially feasible
        assert solver.solve().status == "OPTIMAL"

        saboteur = SaboteurAgent(solver, seed=42)
        result = saboteur.inject_type_d()

        assert result.error_type == ErrorType.TYPE_D
        assert "_saboteur_conflict_" in result.target_name
        assert "Remove contradicting constraint" in result.ground_truth_fix
        assert result.success  # Type D should always cause infeasibility

    def test_type_d_causes_infeasibility(self):
        """Test that Type D always causes infeasibility."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)

        saboteur = SaboteurAgent(solver, seed=42)
        result = saboteur.inject_type_d()

        assert result.solver_status in ["INFEASIBLE", "INF_OR_UNBD"]


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_inject_error_by_type(self):
        """Test inject_error with type string."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent, ErrorType

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)
        saboteur = SaboteurAgent(solver, seed=42)

        result = saboteur.inject_error("A")
        assert result.error_type == ErrorType.TYPE_A

    def test_inject_error_invalid_type(self):
        """Test inject_error with invalid type raises error."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)
        saboteur = SaboteurAgent(solver)

        with pytest.raises(ValueError):
            saboteur.inject_error("X")

    def test_validate_injection(self):
        """Test validate_injection method."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)
        saboteur = SaboteurAgent(solver, seed=42)

        # Before injection, should be valid (OPTIMAL)
        assert not saboteur.validate_injection()

        # After Type D injection, should be invalid
        saboteur.inject_type_d()
        assert saboteur.validate_injection()

    def test_injection_history(self):
        """Test injection history tracking."""
        from src.solvers import GurobiSolver
        from src.data_generation import SaboteurAgent

        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)
        saboteur = SaboteurAgent(solver, seed=42)

        assert len(saboteur.get_injection_history()) == 0
        assert saboteur.get_last_injection() is None

        saboteur.inject_type_d()

        assert len(saboteur.get_injection_history()) == 1
        assert saboteur.get_last_injection() is not None
