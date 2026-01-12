#!/usr/bin/env python3
"""
Tutorial 01: Basic Debugging with OR-Debug-Bench

This tutorial covers:
1. Understanding MDP State and Actions
2. Manual debugging step-by-step
3. Using different agent types
4. Understanding rewards and evaluation

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/

Prerequisites:
    - Gurobi installed and licensed
    - Python packages: gurobipy

Usage:
    python demo/tutorials/01_basic_debugging.py
"""

import gurobipy as gp

from src.solvers import GurobiSolver
from src.environments import (
    SolverDebugEnv,
    DebugState,
    Action,
    ActionType,
)
from src.agents import (
    HeuristicAgent,
    RandomAgent,
    GreedyDropAgent,
    DoNothingAgent,
)
from src.evaluation import (
    MetricsCalculator,
    EpisodeResult,
)


def section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


# =============================================================================
# Part 1: Understanding the MDP State
# =============================================================================


def tutorial_mdp_state():
    """Learn about MDP state representation."""
    section("Part 1: Understanding MDP State")

    # Create a simple infeasible model
    m = gp.Model("tutorial")
    m.Params.OutputFlag = 0

    x = m.addVar(lb=0, ub=10, name="x")
    y = m.addVar(lb=0, ub=10, name="y")

    # Conflicting constraints: x >= 8 AND x <= 5
    m.addConstr(x >= 8, name="x_lower")
    m.addConstr(x <= 5, name="x_upper")
    m.addConstr(x + y <= 15, name="capacity")

    m.setObjective(x + y, gp.GRB.MAXIMIZE)
    m.update()

    solver = GurobiSolver.from_model(m)
    env = SolverDebugEnv(solver)

    state, info = env.reset()

    subsection("State Components")

    print("The DebugState contains:")
    print(f"  1. solver_status: {state.solver_status}")
    print(f"  2. constraint_names: {state.constraint_names}")
    print(f"  3. variable_names: {state.variable_names}")
    print(f"  4. iis_constraints: {state.iis_constraints}")
    print(f"  5. iis_bounds: {state.iis_bounds}")
    print(f"  6. step_count: {state.step_count}")
    print(f"  7. objective: {state.objective}")

    subsection("State Helper Methods")

    print("Useful state methods:")
    print(f"  state.is_optimal(): {state.is_optimal()}")
    print(f"  state.is_infeasible(): {state.is_infeasible()}")
    print(f"  state.get_iis_size(): {state.get_iis_size()}")

    return state, env


# =============================================================================
# Part 2: Understanding Actions
# =============================================================================


def tutorial_actions(env):
    """Learn about available actions."""
    section("Part 2: Understanding Actions")

    subsection("Action Types")

    print("Available action types:")
    print("  Diagnosis Actions:")
    print("    - GET_IIS: Compute Irreducible Infeasible Subsystem")
    print("    - CHECK_SLACK: Check slack value of a constraint")
    print()
    print("  Repair Actions:")
    print("    - DROP_CONSTRAINT: Remove a constraint")
    print("    - RELAX_CONSTRAINT: Relax constraint bounds")
    print("    - UPDATE_RHS: Update right-hand side value")
    print("    - UPDATE_BOUNDS: Update variable bounds")
    print()
    print("  Meta Actions:")
    print("    - RESET: Reset model to original state")
    print("    - SUBMIT: Submit current model as solution")

    subsection("Creating Actions")

    # Examples of creating actions
    action1 = Action(ActionType.GET_IIS)
    action2 = Action(ActionType.DROP_CONSTRAINT, target="x_lower")
    action3 = Action(ActionType.UPDATE_RHS, target="capacity", value=20.0)

    print("Action examples:")
    print(f"  Action(ActionType.GET_IIS) -> {action1}")
    print(f"  Action(ActionType.DROP_CONSTRAINT, target='x_lower') -> {action2}")
    print(f"  Action(ActionType.UPDATE_RHS, target='capacity', value=20) -> {action3}")

    subsection("Using Helper Functions")

    from src.environments import get_iis, drop_constraint, submit

    print("Convenience functions:")
    print(f"  get_iis() -> {get_iis()}")
    print(f"  drop_constraint('c1') -> {drop_constraint('c1')}")
    print(f"  submit() -> {submit()}")


# =============================================================================
# Part 3: Manual Debugging
# =============================================================================


def tutorial_manual_debugging():
    """Learn to debug step-by-step."""
    section("Part 3: Manual Debugging")

    # Create infeasible model
    m = gp.Model("manual_debug")
    m.Params.OutputFlag = 0

    x = m.addVar(lb=0, ub=10, name="x")
    m.addConstr(x >= 8, name="x_lower")
    m.addConstr(x <= 5, name="x_upper")
    m.setObjective(x, gp.GRB.MAXIMIZE)
    m.update()

    solver = GurobiSolver.from_model(m)
    env = SolverDebugEnv(solver, max_steps=10)

    state, _ = env.reset()

    subsection("Initial State")
    print(f"Status: {state.solver_status}")
    print(f"Constraints: {state.constraint_names}")
    print(f"IIS (before computing): {state.iis_constraints}")

    subsection("Step 1: GET_IIS")
    action = Action(ActionType.GET_IIS)
    state, reward, terminated, truncated, info = env.step(action)

    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"IIS (after computing): {state.iis_constraints}")
    print(f"Status: {state.solver_status}")

    subsection("Step 2: DROP_CONSTRAINT")

    # Drop first IIS constraint
    target = state.iis_constraints[0]
    action = Action(ActionType.DROP_CONSTRAINT, target=target)
    state, reward, terminated, truncated, info = env.step(action)

    print(f"Action: {action}")
    print(f"Dropped: {target}")
    print(f"Reward: {reward}")
    print(f"Status: {state.solver_status}")
    print(f"Terminated: {terminated}")

    subsection("Step 3: SUBMIT")

    if not terminated:
        action = Action(ActionType.SUBMIT)
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Final Status: {state.solver_status}")

    print(f"\nDebug successful: {state.is_optimal()}")


# =============================================================================
# Part 4: Using Different Agents
# =============================================================================


def tutorial_agents():
    """Compare different agent types."""
    section("Part 4: Using Different Agents")

    def create_infeasible_env():
        m = gp.Model()
        m.Params.OutputFlag = 0
        x = m.addVar(lb=0, ub=10, name="x")
        m.addConstr(x >= 8, name="x_lower")
        m.addConstr(x <= 5, name="x_upper")
        m.setObjective(x, gp.GRB.MAXIMIZE)
        m.update()
        solver = GurobiSolver.from_model(m)
        return SolverDebugEnv(solver, max_steps=10)

    agents = [
        ("HeuristicAgent", HeuristicAgent()),
        ("RandomAgent", RandomAgent(seed=42)),
        ("GreedyDropAgent", GreedyDropAgent()),
        ("DoNothingAgent", DoNothingAgent()),
    ]

    results = []

    for name, agent in agents:
        subsection(f"Testing {name}")

        env = create_infeasible_env()
        state, _ = env.reset()
        agent.reset()

        total_reward = 0
        steps = 0

        while not env.is_done and steps < 10:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            print(f"  Step {steps}: {action.action_type.value}", end="")
            if action.target:
                print(f" ({action.target})", end="")
            print(f" -> R={reward:.1f}")

            if terminated or truncated:
                break

        success = state.is_optimal()
        print(f"\n  Result: {'Success' if success else 'Failed'}")
        print(f"  Steps: {steps}, Total Reward: {total_reward:.1f}")

        results.append(EpisodeResult(
            success=success,
            final_status=state.solver_status,
            steps=steps,
            total_reward=total_reward,
            agent_name=name,
        ))

    # Compare results
    subsection("Agent Comparison")

    calc = MetricsCalculator()
    print(f"{'Agent':<20} {'Success':<10} {'Steps':<10} {'Reward':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r.agent_name:<20} {str(r.success):<10} {r.steps:<10} {r.total_reward:<10.1f}")


# =============================================================================
# Part 5: Understanding Rewards
# =============================================================================


def tutorial_rewards():
    """Learn about the reward structure."""
    section("Part 5: Understanding Rewards")

    print("Reward Structure:")
    print()
    print("  Outcome Rewards:")
    print("    +100  : Successfully reached OPTIMAL status")
    print("    -50   : SUBMIT while still INFEASIBLE")
    print()
    print("  Process Rewards:")
    print("    +10   : IIS size reduction (per constraint removed)")
    print("    -1    : Step penalty (each action costs -1)")
    print()
    print("  Faithfulness Penalty:")
    print("    -20   : Diagnosis contradicts solver logs (future)")

    subsection("Reward Example")

    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVar(lb=0, ub=10, name="x")
    m.addConstr(x >= 8, name="x_lower")
    m.addConstr(x <= 5, name="x_upper")
    m.setObjective(x, gp.GRB.MAXIMIZE)
    m.update()

    solver = GurobiSolver.from_model(m)
    env = SolverDebugEnv(solver, max_steps=10)

    state, _ = env.reset()
    total_reward = 0

    # Step 1: GET_IIS
    action = Action(ActionType.GET_IIS)
    state, reward, _, _, _ = env.step(action)
    print(f"GET_IIS:         Reward = {reward:+.1f} (step penalty)")
    total_reward += reward

    # Step 2: DROP_CONSTRAINT (removes from IIS)
    action = Action(ActionType.DROP_CONSTRAINT, target=state.iis_constraints[0])
    state, reward, terminated, _, _ = env.step(action)
    print(f"DROP_CONSTRAINT: Reward = {reward:+.1f} (IIS reduction + step penalty + outcome)")
    total_reward += reward

    print(f"\nTotal Episode Reward: {total_reward:.1f}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all tutorial sections."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  Tutorial 01: Basic Debugging with OR-Debug-Bench  ".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    # Part 1: MDP State
    state, env = tutorial_mdp_state()

    # Part 2: Actions
    tutorial_actions(env)

    # Part 3: Manual debugging
    tutorial_manual_debugging()

    # Part 4: Different agents
    tutorial_agents()

    # Part 5: Rewards
    tutorial_rewards()

    # Summary
    section("Tutorial Complete!")

    print("You have learned:")
    print("  1. DebugState structure and helper methods")
    print("  2. Available action types and how to create them")
    print("  3. Manual step-by-step debugging")
    print("  4. Using different agent types")
    print("  5. Reward structure and calculation")
    print()
    print("Next Steps:")
    print("  - Try demo/quickstart.py for end-to-end example")
    print("  - Explore src/agents/llm_agent.py for LLM integration")
    print("  - Read docs/directions/A_OR_Debug_Bench/ for full spec")


if __name__ == "__main__":
    main()
