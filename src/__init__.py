"""
LLM-for-OR: Agentic Operations Research with Large Language Models

This package provides evaluation benchmarks and training infrastructure
for LLM agents in Operations Research and Operations Management.

Research Direction: Direction A (OR-Debug-Bench) - Primary
Documentation: docs/directions/A_OR_Debug_Bench/

Key Components:
    - environments: MDP environments wrapping optimization solvers
    - agents: LLM and RL agent implementations
    - solvers: Gurobi/Pyomo interfaces with IIS extraction
    - data_generation: Saboteur agent for error injection
    - evaluation: Metrics (Recovery Rate, Diagnosis Accuracy, etc.)
"""

__version__ = "0.1.0"
__author__ = "Ruicheng Ao"
