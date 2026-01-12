"""
Unit tests for GurobiSolver interface.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/

Test Coverage:
    - Model creation and loading
    - Solve and status extraction
    - IIS computation for infeasible models
    - Constraint/variable information extraction
    - Model modification operations
    - Clone and reset functionality
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


class TestGurobiSolverCreation:
    """Tests for GurobiSolver creation and initialization."""

    def test_create_empty_solver(self):
        """Test creating solver without model."""
        from src.solvers import GurobiSolver

        solver = GurobiSolver()
        assert solver is not None
        assert solver.get_num_constraints() == 0
        assert solver.get_num_variables() == 0

    def test_create_from_model(self):
        """Test creating solver from existing Gurobi model."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        # Create a simple LP
        model = gp.Model("test")
        x = model.addVar(name="x", lb=0, ub=10)
        y = model.addVar(name="y", lb=0, ub=10)
        model.addConstr(x + y <= 15, name="c1")
        model.setObjective(x + 2 * y, gp.GRB.MAXIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        assert solver.get_num_variables() == 2
        assert solver.get_num_constraints() == 1


class TestGurobiSolverSolve:
    """Tests for solving models and extracting state."""

    def test_solve_optimal(self):
        """Test solving a feasible LP."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("feasible")
        x = model.addVar(name="x", lb=0, ub=10)
        y = model.addVar(name="y", lb=0, ub=10)
        model.addConstr(x + y <= 15, name="c1")
        model.setObjective(x + 2 * y, gp.GRB.MAXIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        state = solver.solve()

        assert state.status == "OPTIMAL"
        assert state.objective is not None
        assert state.objective == pytest.approx(30.0)  # x=0, y=10
        assert state.solve_time >= 0

    def test_solve_infeasible(self):
        """Test solving an infeasible model."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("infeasible")
        x = model.addVar(name="x", lb=0)
        model.addConstr(x >= 10, name="c1")
        model.addConstr(x <= 5, name="c2")  # Conflict!
        model.setObjective(x, gp.GRB.MINIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        state = solver.solve()

        assert state.status == "INFEASIBLE"
        assert state.objective is None

    def test_solve_unbounded(self):
        """Test solving an unbounded model."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("unbounded")
        x = model.addVar(name="x", lb=0)  # No upper bound
        model.setObjective(x, gp.GRB.MAXIMIZE)  # Maximize unbounded var
        model.update()

        solver = GurobiSolver.from_model(model)
        state = solver.solve()

        assert state.status in ["UNBOUNDED", "INF_OR_UNBD"]


class TestGurobiSolverIIS:
    """Tests for IIS computation."""

    def test_compute_iis(self):
        """Test IIS extraction for infeasible model."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("iis_test")
        x = model.addVar(name="x", lb=0)
        model.addConstr(x >= 10, name="lower")
        model.addConstr(x <= 5, name="upper")  # Conflict with lower
        model.setObjective(x, gp.GRB.MINIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        state = solver.solve()
        assert state.status == "INFEASIBLE"

        iis = solver.compute_iis()
        assert iis.size > 0
        # Both constraints should be in IIS
        assert "lower" in iis.constraints or "upper" in iis.constraints

    def test_compute_iis_error_on_optimal(self):
        """Test that IIS computation fails for optimal model."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("optimal")
        x = model.addVar(name="x", lb=0, ub=10)
        model.setObjective(x, gp.GRB.MINIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        state = solver.solve()
        assert state.status == "OPTIMAL"

        with pytest.raises(RuntimeError):
            solver.compute_iis()


class TestGurobiSolverInfo:
    """Tests for constraint and variable information extraction."""

    def test_get_constraint_info(self):
        """Test constraint info extraction."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("info_test")
        x = model.addVar(name="x", lb=0, ub=10)
        model.addConstr(x <= 5, name="limit")
        model.setObjective(x, gp.GRB.MAXIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        solver.solve()

        info = solver.get_constraint_info("limit")
        assert info.name == "limit"
        assert info.sense == "<="
        assert info.rhs == 5.0
        assert info.slack is not None

    def test_get_variable_info(self):
        """Test variable info extraction."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("var_info_test")
        x = model.addVar(name="x", lb=0, ub=10, vtype=gp.GRB.BINARY)
        model.setObjective(x, gp.GRB.MAXIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        solver.solve()

        info = solver.get_variable_info("x")
        assert info.name == "x"
        assert info.vtype == "B"  # Binary
        assert info.lb == 0
        assert info.ub == 1  # Binary has ub=1
        assert info.value is not None

    def test_get_all_constraints(self):
        """Test getting all constraint names."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("multi_constr")
        x = model.addVar(name="x")
        model.addConstr(x >= 0, name="c1")
        model.addConstr(x <= 10, name="c2")
        model.addConstr(x >= 5, name="c3")
        model.update()

        solver = GurobiSolver.from_model(model)
        names = solver.get_all_constraints()

        assert len(names) == 3
        assert set(names) == {"c1", "c2", "c3"}


class TestGurobiSolverModification:
    """Tests for model modification operations."""

    def test_relax_constraint(self):
        """Test constraint relaxation."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        # Create infeasible model
        model = gp.Model("relax_test")
        x = model.addVar(name="x", lb=0)
        model.addConstr(x >= 10, name="lower")
        model.addConstr(x <= 5, name="upper")
        model.setObjective(x, gp.GRB.MINIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)

        # Verify infeasible
        state = solver.solve()
        assert state.status == "INFEASIBLE"

        # Relax upper constraint
        solver.relax_constraint("upper", 10)  # upper becomes x <= 15

        # Should now be feasible
        state = solver.solve()
        assert state.status == "OPTIMAL"

    def test_drop_constraint(self):
        """Test constraint removal."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("drop_test")
        x = model.addVar(name="x", lb=0)
        model.addConstr(x >= 10, name="lower")
        model.addConstr(x <= 5, name="upper")
        model.setObjective(x, gp.GRB.MINIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)

        # Verify infeasible
        state = solver.solve()
        assert state.status == "INFEASIBLE"

        # Drop conflicting constraint
        solver.drop_constraint("lower")

        # Should now be feasible
        state = solver.solve()
        assert state.status == "OPTIMAL"
        assert solver.get_num_constraints() == 1

    def test_update_rhs(self):
        """Test RHS update."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("rhs_test")
        x = model.addVar(name="x", lb=0)
        model.addConstr(x <= 5, name="limit")
        model.setObjective(x, gp.GRB.MAXIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        state = solver.solve()
        assert state.objective == pytest.approx(5.0)

        # Update RHS
        solver.update_rhs("limit", 10)
        state = solver.solve()
        assert state.objective == pytest.approx(10.0)


class TestGurobiSolverStateManagement:
    """Tests for clone and reset functionality."""

    def test_clone(self):
        """Test model cloning."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("clone_test")
        x = model.addVar(name="x", lb=0, ub=10)
        model.addConstr(x <= 5, name="limit")
        model.setObjective(x, gp.GRB.MAXIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)
        clone = solver.clone()

        # Modify original
        solver.update_rhs("limit", 3)

        # Clone should be unchanged
        solver.solve()
        clone.solve()

        orig_info = solver.get_constraint_info("limit")
        clone_info = clone.get_constraint_info("limit")

        assert orig_info.rhs == 3
        assert clone_info.rhs == 5

    def test_reset(self):
        """Test model reset."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("reset_test")
        x = model.addVar(name="x", lb=0, ub=10)
        model.addConstr(x <= 5, name="limit")
        model.setObjective(x, gp.GRB.MAXIMIZE)
        model.update()

        solver = GurobiSolver.from_model(model)

        # Modify
        solver.update_rhs("limit", 3)
        solver.solve()
        assert solver.get_constraint_info("limit").rhs == 3

        # Reset
        solver.reset()
        solver.solve()
        assert solver.get_constraint_info("limit").rhs == 5

    def test_get_state_snapshot(self):
        """Test state snapshot extraction."""
        from src.solvers import GurobiSolver
        import gurobipy as gp

        model = gp.Model("snapshot_test")
        x = model.addVar(name="x", lb=0, ub=10)
        model.addConstr(x <= 5, name="limit")
        model.update()

        solver = GurobiSolver.from_model(model)
        snapshot = solver.get_state_snapshot()

        assert "constraints" in snapshot
        assert "variables" in snapshot
        assert "limit" in snapshot["constraints"]
        assert "x" in snapshot["variables"]
        assert snapshot["constraints"]["limit"]["rhs"] == 5
        assert snapshot["variables"]["x"]["lb"] == 0
        assert snapshot["variables"]["x"]["ub"] == 10
