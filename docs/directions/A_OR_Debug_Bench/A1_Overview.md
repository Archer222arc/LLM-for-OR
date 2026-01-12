# Direction A: OR-Debug-Bench — Overview

## Why Self-Correction Evaluation Wins

### Argument 1: Specificity (The "GNN Defense")

In OR-Debug-Bench, state includes natural language error logs (e.g., "Model is infeasible. IIS indicates conflict between Constraint A and Constraint B"). Interpreting these semantic logs, mapping them to original problems, and generating Python fixes fundamentally requires LLM capabilities. GNNs operate on fixed graph structures—they cannot process semantic error messages or rewrite executable code. **LLM's role is unchallengeable.**

### Argument 2: MDP Novelty (Process vs. Outcome)

Existing benchmarks measure `P(correct code | text)`. OR-Debug-Bench measures policy `π(aₜ | sₜ)` where agents navigate erroneous states to find feasible regions. This enables evaluation of:
- **Efficiency**: Steps to fix
- **Robustness**: Does fix break other constraints?
- **Reasoning Quality**: Diagnosis accuracy

Metrics shift from "Pass@1" to **Recovery Rate**, **Trajectory Efficiency**, and **Feasibility Preservation**.

### Argument 3: Literature Validation

| Source | Validation |
|--------|------------|
| **CorrectBench** | Documents 64.5% blind spot rate in self-correction—massive unaddressed vulnerability in OR where single constraint errors break systems |
| **MOID** | Explicitly validates LLMs can perform infeasibility diagnosis, generating constraint-aware heuristics and Pareto fronts of "least-violating" solutions |
| **StepORLM** | Provides blueprint for process supervision in OR; OR-Debug-Bench extends this to the debugging phase |
| **Gap Identification** | CorrectBench omits constrained optimization entirely—creating "blue ocean" opportunity for OR-specific contribution |

---

## Core Concept

**OR-Debug-Bench** is a Gym-like MDP environment for evaluating LLM agents' ability to:

1. **Diagnose** solver infeasibility/unboundedness errors
2. **Analyze** IIS (Irreducible Infeasible Subsystem) logs
3. **Recover** feasibility through targeted constraint modifications
4. **Preserve** solution quality (not over-relax the model)

Unlike static benchmarks that test one-shot formulation accuracy, OR-Debug-Bench tests the **iterative debugging process** that real practitioners perform daily.

---

## The OR Advantage for Self-Correction

> Unlike creative writing where "correctness" is subjective, an optimization solver returns precise, incontrovertible feedback. Gurobi generates an IIS isolating exact conflicting constraints. This transforms self-correction from "introspective guessing" to "diagnostic reasoning."

| Domain | Feedback Type | Self-Correction Approach |
|--------|---------------|-------------------------|
| Creative Writing | Subjective | Introspective guessing |
| Code Generation | Test cases | Binary pass/fail |
| **Mathematical Optimization** | **IIS + Solver Status** | **Diagnostic reasoning with ground truth** |

---

*Next: [A2_MDP_Spec.md](A2_MDP_Spec.md) - Formal MDP Specification*
