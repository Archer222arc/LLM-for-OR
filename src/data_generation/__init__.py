"""
Data generation module for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A3_Data_Generation.md

Key Components:
    - SaboteurAgent: Error injection agent with robust methods
    - ErrorType: Classification of error types (A-D)
    - Difficulty: Difficulty classification (easy/medium/hard)
    - InjectionResult: Result of error injection
    - ProblemValidator: Four-phase validation pipeline

Example:
    >>> from src.data_generation import SaboteurAgent, ErrorType, ProblemValidator
    >>> from src.solvers import GurobiSolver
    >>> solver = GurobiSolver.from_file("model.mps")
    >>> saboteur = SaboteurAgent(solver)
    >>> result = saboteur.inject_type_a_robust()  # Use robust method
    >>> print(f"Success: {result.success}, Difficulty: {result.difficulty}")
"""

from .error_types import ErrorType, InjectionResult, Difficulty
from .saboteur_agent import SaboteurAgent
from .validator import ProblemValidator, ValidationResult, validate_dataset

__all__ = [
    "ErrorType",
    "Difficulty",
    "InjectionResult",
    "SaboteurAgent",
    "ProblemValidator",
    "ValidationResult",
    "validate_dataset",
]
