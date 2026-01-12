#!/usr/bin/env python3
"""
OR-Debug-Bench Quick Start

This script demonstrates the core functionality of OR-Debug-Bench in 5 minutes:
1. Create an infeasible optimization model
2. Use SaboteurAgent to inject errors (data generation)
3. Use SolverDebugEnv (MDP environment)
4. Use HeuristicAgent to debug the model
5. Evaluate results with MetricsCalculator

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/

Usage:
    python demo/quickstart.py
"""

import gurobipy as gp

from src.solvers import GurobiSolver
from src.data_generation import SaboteurAgent
from src.environments import SolverDebugEnv, ActionType
from src.agents import HeuristicAgent, RandomAgent
from src.evaluation import (
    BenchmarkRunner,
    BenchmarkProblem,
    BenchmarkConfig,
    MetricsCalculator,
)


def create_feasible_model():
    """
    Create a simple feasible optimization model.

    Problem: Maximize x + y
    Subject to:
        x + y <= 10
        x >= 0, y >= 0
        x <= 8, y <= 8
    """
    m = gp.Model("feasible_model")
    m.Params.OutputFlag = 0  # Suppress Gurobi output

    # Variables
    x = m.addVar(lb=0, ub=8, name="x")
    y = m.addVar(lb=0, ub=8, name="y")

    # Constraints
    m.addConstr(x + y <= 10, name="capacity")

    # Objective
    m.setObjective(x + y, gp.GRB.MAXIMIZE)
    m.update()

    return m


def demo_basic_debugging():
    """Demonstrate basic debugging workflow."""
    print("=" * 60)
    print("OR-Debug-Bench Quick Start Demo")
    print("=" * 60)

    # Step 1: Create a feasible model
    print("\n[Step 1] Creating a feasible optimization model...")
    model = create_feasible_model()
    solver = GurobiSolver.from_model(model)

    # Verify it's initially feasible
    state = solver.solve()
    print(f"  Initial status: {state.status}")
    print(f"  Objective value: {state.objective}")

    # Step 2: Inject an error using SaboteurAgent
    print("\n[Step 2] Injecting error using SaboteurAgent...")
    saboteur = SaboteurAgent(solver, seed=42)
    result = saboteur.inject_type_d()  # Add conflicting constraint

    print(f"  Error type: {result.error_type}")
    print(f"  Target: {result.target_name}")
    print(f"  New status: {result.solver_status}")

    # Step 3: Create MDP environment
    print("\n[Step 3] Creating SolverDebugEnv (MDP environment)...")
    env = SolverDebugEnv(solver, max_steps=10)
    state, info = env.reset()

    print(f"  Environment status: {state.solver_status}")
    print(f"  Constraints: {state.constraint_names}")

    # Step 4: Use HeuristicAgent to debug
    print("\n[Step 4] Using HeuristicAgent to debug...")
    agent = HeuristicAgent()

    step = 0
    total_reward = 0

    while not env.is_done and step < 10:
        action = agent.act(state)
        print(f"  Step {step + 1}: {action.action_type.value}", end="")
        if action.target:
            print(f" ({action.target})", end="")

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f" -> Status: {state.solver_status}, Reward: {reward:.1f}")

        step += 1

        if terminated or truncated:
            break

    # Step 5: Evaluate results
    print("\n[Step 5] Evaluation Results")
    print("-" * 40)
    success = state.is_optimal()
    print(f"  Success: {success}")
    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.1f}")

    return success, step, total_reward


def demo_benchmark_comparison():
    """Demonstrate benchmark comparison between agents."""
    print("\n" + "=" * 60)
    print("Agent Comparison Demo")
    print("=" * 60)

    # Create multiple problem instances
    problems = []
    for i in range(3):
        model = create_feasible_model()
        solver = GurobiSolver.from_model(model)

        # Inject error
        saboteur = SaboteurAgent(solver, seed=i)
        saboteur.inject_type_d()

        env = SolverDebugEnv(solver, max_steps=10)
        problems.append(BenchmarkProblem(
            problem_id=f"problem_{i}",
            env=env,
        ))

    # Compare agents
    config = BenchmarkConfig(max_steps=10, n_episodes=1, verbose=False)
    runner = BenchmarkRunner(config=config)

    agents = [
        HeuristicAgent(),
        RandomAgent(seed=42),
    ]

    print("\nRunning benchmark...")
    comparison = runner.compare_agents(problems, agents)

    # Print comparison
    print("\n" + runner.format_comparison(comparison))

    return comparison


def demo_metrics():
    """Demonstrate metrics calculation."""
    print("\n" + "=" * 60)
    print("Metrics Calculation Demo")
    print("=" * 60)

    # Run a benchmark
    model = create_feasible_model()
    solver = GurobiSolver.from_model(model)
    saboteur = SaboteurAgent(solver, seed=42)
    saboteur.inject_type_d()

    env = SolverDebugEnv(solver, max_steps=10)
    problems = [BenchmarkProblem(problem_id="demo", env=env)]

    runner = BenchmarkRunner()
    runner.run_benchmark(problems, HeuristicAgent())

    # Get and print summary
    print("\n" + runner.format_summary())

    return runner.get_summary()


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  OR-Debug-Bench: LLM Agent for Solver Debugging  ".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    # Demo 1: Basic debugging
    success, steps, reward = demo_basic_debugging()

    # Demo 2: Benchmark comparison
    comparison = demo_benchmark_comparison()

    # Demo 3: Metrics
    summary = demo_metrics()

    # Final summary
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. SaboteurAgent injects controlled errors into feasible models")
    print("  2. SolverDebugEnv provides Gymnasium-style MDP interface")
    print("  3. Agents (Heuristic, Random, LLM) can interact with the env")
    print("  4. BenchmarkRunner enables systematic evaluation")
    print("  5. MetricsCalculator provides recovery rate, steps, rewards")
    print("\nNext Steps:")
    print("  - Try LLMAgent with GPT-4 or Claude")
    print("  - Run on MIPLIB instances")
    print("  - Train RL agents using GRPO")
    print("=" * 60)


if __name__ == "__main__":
    main()
