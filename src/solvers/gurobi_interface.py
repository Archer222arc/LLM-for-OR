"""
Gurobi solver interface for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/

Key Components:
    - GurobiSolver: Main solver class wrapping gurobipy
    - Status extraction and mapping
    - IIS computation and parsing
    - Model modification operations

Example:
    >>> from src.solvers import GurobiSolver
    >>> solver = GurobiSolver.from_file("model.mps")
    >>> state = solver.solve()
    >>> if state.status == "INFEASIBLE":
    ...     iis = solver.compute_iis()
    ...     print(f"Conflicting constraints: {iis.constraints}")
"""

import gurobipy as gp
from typing import List, Optional, Dict, Any
from pathlib import Path

from .base_solver import (
    BaseSolver,
    SolverState,
    IISResult,
    ConstraintInfo,
    VariableInfo,
)


# Gurobi status code to string mapping
STATUS_MAP = {
    gp.GRB.LOADED: "LOADED",
    gp.GRB.OPTIMAL: "OPTIMAL",
    gp.GRB.INFEASIBLE: "INFEASIBLE",
    gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
    gp.GRB.UNBOUNDED: "UNBOUNDED",
    gp.GRB.CUTOFF: "CUTOFF",
    gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    gp.GRB.NODE_LIMIT: "NODE_LIMIT",
    gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    gp.GRB.INTERRUPTED: "INTERRUPTED",
    gp.GRB.NUMERIC: "NUMERIC",
    gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    gp.GRB.INPROGRESS: "INPROGRESS",
    gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}

# Constraint sense mapping
SENSE_MAP = {
    gp.GRB.LESS_EQUAL: "<=",
    gp.GRB.GREATER_EQUAL: ">=",
    gp.GRB.EQUAL: "=",
}

# Variable type mapping
VTYPE_MAP = {
    gp.GRB.CONTINUOUS: "C",
    gp.GRB.BINARY: "B",
    gp.GRB.INTEGER: "I",
    gp.GRB.SEMICONT: "S",
    gp.GRB.SEMIINT: "N",
}


class GurobiSolver(BaseSolver):
    """
    Gurobi solver wrapper for OR-Debug-Bench MDP environment.

    Provides a clean interface for:
    - Solving optimization models
    - Extracting solver state and IIS
    - Modifying constraints for repair actions
    - Cloning/resetting for episode management

    Attributes:
        _model: The current Gurobi model
        _original_model: Copy of the original model for reset
        _iis_computed: Whether IIS has been computed for current state
    """

    def __init__(self, model: Optional[gp.Model] = None, name: str = "model"):
        """
        Initialize GurobiSolver.

        Args:
            model: Optional Gurobi model. If None, creates empty model.
            name: Model name (used if creating new model).
        """
        if model is not None:
            self._model = model
            self._original_model = model.copy()
        else:
            self._model = gp.Model(name)
            self._original_model = None

        self._iis_computed = False

        # Suppress Gurobi output by default
        self._model.Params.OutputFlag = 0

    @classmethod
    def from_file(cls, path: str) -> 'GurobiSolver':
        """
        Load a model from file.

        Args:
            path: Path to model file (.mps, .lp, .mst, etc.)

        Returns:
            GurobiSolver instance with loaded model.
        """
        model = gp.read(str(path))
        return cls(model=model)

    @classmethod
    def from_model(cls, model: gp.Model) -> 'GurobiSolver':
        """
        Create solver from existing Gurobi model.

        Args:
            model: Gurobi Model object.

        Returns:
            GurobiSolver instance wrapping the model.
        """
        return cls(model=model)

    # =========================================================================
    # Core Solving Methods
    # =========================================================================

    def solve(self) -> SolverState:
        """
        Solve the optimization model.

        Returns:
            SolverState with status, objective value, and metrics.
        """
        self._iis_computed = False  # Reset IIS flag
        self._model.optimize()

        status_code = self._model.Status
        status = STATUS_MAP.get(status_code, "ERROR")

        # Extract objective if optimal
        objective = None
        if status == "OPTIMAL":
            objective = self._model.ObjVal

        # Extract MIP gap if available
        gap = None
        if self._model.IsMIP and status == "OPTIMAL":
            gap = self._model.MIPGap

        # Extract solve metrics
        solve_time = self._model.Runtime
        node_count = int(self._model.NodeCount) if self._model.IsMIP else 0
        iteration_count = int(self._model.IterCount)

        return SolverState(
            status=status,
            objective=objective,
            gap=gap,
            solve_time=solve_time,
            node_count=node_count,
            iteration_count=iteration_count,
        )

    def get_status(self) -> str:
        """Get current solver status as string."""
        return STATUS_MAP.get(self._model.Status, "ERROR")

    # =========================================================================
    # IIS Computation
    # =========================================================================

    def compute_iis(self) -> IISResult:
        """
        Compute Irreducible Infeasible Subsystem.

        Should only be called when model is INFEASIBLE.

        Returns:
            IISResult with conflicting constraints and variable bounds.

        Raises:
            RuntimeError: If model is not infeasible.
        """
        if self._model.Status != gp.GRB.INFEASIBLE:
            raise RuntimeError(
                f"Cannot compute IIS: model status is {self.get_status()}, "
                "expected INFEASIBLE"
            )

        self._model.computeIIS()
        self._iis_computed = True

        # Extract constraint IIS
        constraints = [
            c.ConstrName
            for c in self._model.getConstrs()
            if c.IISConstr
        ]

        # Extract variable bound IIS
        bounds = [
            v.VarName
            for v in self._model.getVars()
            if v.IISLB or v.IISUB
        ]

        return IISResult(constraints=constraints, bounds=bounds)

    # =========================================================================
    # Information Extraction
    # =========================================================================

    def get_constraint_info(self, name: str) -> ConstraintInfo:
        """
        Get detailed information about a constraint.

        Args:
            name: Constraint name.

        Returns:
            ConstraintInfo with sense, RHS, slack, and IIS membership.
        """
        constr = self._model.getConstrByName(name)
        if constr is None:
            raise ValueError(f"Constraint '{name}' not found")

        # Get slack if model was solved
        slack = None
        if self._model.Status == gp.GRB.OPTIMAL:
            slack = constr.Slack

        # Check IIS membership if computed
        is_in_iis = False
        if self._iis_computed:
            is_in_iis = constr.IISConstr

        return ConstraintInfo(
            name=name,
            sense=SENSE_MAP.get(constr.Sense, "?"),
            rhs=constr.RHS,
            slack=slack,
            is_in_iis=is_in_iis,
        )

    def get_variable_info(self, name: str) -> VariableInfo:
        """
        Get detailed information about a variable.

        Args:
            name: Variable name.

        Returns:
            VariableInfo with type, bounds, value, and IIS membership.
        """
        var = self._model.getVarByName(name)
        if var is None:
            raise ValueError(f"Variable '{name}' not found")

        # Get value if model was solved
        value = None
        if self._model.Status == gp.GRB.OPTIMAL:
            value = var.X

        # Check IIS membership if computed
        is_lb_in_iis = False
        is_ub_in_iis = False
        if self._iis_computed:
            is_lb_in_iis = var.IISLB
            is_ub_in_iis = var.IISUB

        return VariableInfo(
            name=name,
            vtype=VTYPE_MAP.get(var.VType, "?"),
            lb=var.LB,
            ub=var.UB,
            value=value,
            is_lb_in_iis=is_lb_in_iis,
            is_ub_in_iis=is_ub_in_iis,
        )

    def get_all_constraints(self) -> List[str]:
        """Get names of all constraints."""
        return [c.ConstrName for c in self._model.getConstrs()]

    def get_all_variables(self) -> List[str]:
        """Get names of all variables."""
        return [v.VarName for v in self._model.getVars()]

    def get_num_constraints(self) -> int:
        """Get number of constraints."""
        return self._model.NumConstrs

    def get_num_variables(self) -> int:
        """Get number of variables."""
        return self._model.NumVars

    # =========================================================================
    # Model Modification (Action Support)
    # =========================================================================

    def relax_constraint(self, name: str, epsilon: float) -> None:
        """
        Relax a constraint by modifying its RHS.

        Args:
            name: Constraint name.
            epsilon: Amount to relax (added to RHS for <=, subtracted for >=).
        """
        constr = self._model.getConstrByName(name)
        if constr is None:
            raise ValueError(f"Constraint '{name}' not found")

        sense = constr.Sense
        if sense == gp.GRB.LESS_EQUAL:
            constr.RHS += abs(epsilon)
        elif sense == gp.GRB.GREATER_EQUAL:
            constr.RHS -= abs(epsilon)
        else:  # EQUAL - convert to inequality
            # For equality, we need to decide direction
            # Default: relax both directions by epsilon
            constr.RHS += epsilon

        self._model.update()
        self._iis_computed = False

    def drop_constraint(self, name: str) -> None:
        """
        Remove a constraint from the model.

        Args:
            name: Constraint name to remove.
        """
        constr = self._model.getConstrByName(name)
        if constr is None:
            raise ValueError(f"Constraint '{name}' not found")

        self._model.remove(constr)
        self._model.update()
        self._iis_computed = False

    def update_rhs(self, name: str, new_rhs: float) -> None:
        """
        Update the RHS of a constraint.

        Args:
            name: Constraint name.
            new_rhs: New RHS value.
        """
        constr = self._model.getConstrByName(name)
        if constr is None:
            raise ValueError(f"Constraint '{name}' not found")

        constr.RHS = new_rhs
        self._model.update()
        self._iis_computed = False

    def update_variable_bounds(
        self,
        name: str,
        lb: Optional[float] = None,
        ub: Optional[float] = None
    ) -> None:
        """
        Update variable bounds.

        Args:
            name: Variable name.
            lb: New lower bound (None to keep current).
            ub: New upper bound (None to keep current).
        """
        var = self._model.getVarByName(name)
        if var is None:
            raise ValueError(f"Variable '{name}' not found")

        if lb is not None:
            var.LB = lb
        if ub is not None:
            var.UB = ub

        self._model.update()
        self._iis_computed = False

    # =========================================================================
    # State Management
    # =========================================================================

    def clone(self) -> 'GurobiSolver':
        """
        Create an independent copy of the solver.

        Returns:
            New GurobiSolver with copied model.
        """
        new_model = self._model.copy()
        new_solver = GurobiSolver(model=new_model)
        new_solver._original_model = self._original_model.copy() if self._original_model else None
        return new_solver

    def reset(self) -> None:
        """Reset model to original state."""
        if self._original_model is None:
            raise RuntimeError("No original model stored - cannot reset")

        self._model = self._original_model.copy()
        self._model.Params.OutputFlag = 0
        self._iis_computed = False

    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current model state.

        Returns:
            Dictionary with constraint RHS values and variable bounds.
        """
        snapshot = {
            "constraints": {},
            "variables": {},
        }

        for c in self._model.getConstrs():
            snapshot["constraints"][c.ConstrName] = {
                "sense": SENSE_MAP.get(c.Sense, "?"),
                "rhs": c.RHS,
            }

        for v in self._model.getVars():
            snapshot["variables"][v.VarName] = {
                "vtype": VTYPE_MAP.get(v.VType, "?"),
                "lb": v.LB,
                "ub": v.UB,
            }

        return snapshot

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def model(self) -> gp.Model:
        """Access the underlying Gurobi model."""
        return self._model

    @property
    def name(self) -> str:
        """Get model name."""
        return self._model.ModelName

    @property
    def is_mip(self) -> bool:
        """Check if model is a MIP."""
        return self._model.IsMIP
