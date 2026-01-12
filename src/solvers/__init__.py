"""
Solver interfaces for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/

Key Components:
    - BaseSolver: Abstract base class for solvers
    - GurobiSolver: Gurobi implementation
    - Data classes: SolverState, IISResult, ConstraintInfo, VariableInfo

Example:
    >>> from src.solvers import GurobiSolver
    >>> solver = GurobiSolver.from_file("model.mps")
    >>> state = solver.solve()
    >>> print(state.status)
"""

from .base_solver import (
    BaseSolver,
    SolverState,
    IISResult,
    ConstraintInfo,
    VariableInfo,
)

from .gurobi_interface import GurobiSolver

__all__ = [
    # Base classes
    "BaseSolver",
    # Data classes
    "SolverState",
    "IISResult",
    "ConstraintInfo",
    "VariableInfo",
    # Implementations
    "GurobiSolver",
]
