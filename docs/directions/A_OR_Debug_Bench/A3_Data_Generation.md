# Direction A: OR-Debug-Bench — Data Generation Strategy

## Overview

Leverage OptMATH pipeline and public libraries for synthetic error injection.

---

## Data Generation Pipeline

### Step 1: Seed Generation

Utilize ~200,000 feasible instances from OptMATH or MIPLIB as "Ground Truth":

| Source | Scale | Type |
|--------|-------|------|
| OptMATH | 200k+ pairs | NL + Code |
| MIPLIB 2017 | 1,065 instances | MPS format |
| IndustryOR | 100 cases | Industrial complexity |
| NL4Opt | 1,101 problems | LP/MILP |

---

### Step 2: Systematic Error Injection — The Saboteur Agent

Develop **"Saboteur Agent"** to inject common errors:

| Type | Error Category | Example | Detection |
|------|---------------|---------|-----------|
| **Type A** | Bound Error | Swapping ≤ for ≥ in constraints | Constraint direction flip |
| **Type B** | Variable Error | Defining integer as continuous or vice versa | Variable type mismatch |
| **Type C** | Logic Error | Omitting summation term or index in loop | Missing terms |
| **Type D** | Conflict Error | Adding directly contradicting constraints | Explicit contradiction |

#### Saboteur Agent Implementation

```python
class SaboteurAgent:
    """Injects controlled errors into feasible optimization models."""

    def inject_error(self, model, error_type: str):
        if error_type == "A":  # Bound Error
            constraint = random.choice(model.constraints)
            constraint.sense = flip_sense(constraint.sense)
        elif error_type == "B":  # Variable Error
            var = random.choice(model.integer_vars)
            var.vtype = "continuous"
        elif error_type == "C":  # Logic Error
            constraint = random.choice(model.constraints)
            constraint.expr = remove_random_term(constraint.expr)
        elif error_type == "D":  # Conflict Error
            new_constraint = generate_contradicting_constraint(model)
            model.add_constraint(new_constraint)
        return model
```

---

### Step 3: Validation

Run Gurobi on injected code to confirm target error status:

```python
def validate_injection(original_model, sabotaged_model):
    # Original should be feasible
    assert solve(original_model).status == OPTIMAL

    # Sabotaged should be infeasible/unbounded
    result = solve(sabotaged_model)
    assert result.status in [INFEASIBLE, UNBOUNDED]

    # Record ground truth fix
    ground_truth = {
        "error_type": error_type,
        "error_location": error_location,
        "fix": compute_minimal_fix(original_model, sabotaged_model)
    }
    return ground_truth
```

---

### Step 4: Dataset Creation

Pair `(Original NL Description, Broken Code)` forms initial state s₀:

```json
{
    "instance_id": "miplib_2017_0001_typeA",
    "problem_nl": "Minimize production cost subject to...",
    "original_code": "...",
    "broken_code": "...",
    "error_type": "A",
    "error_location": {"constraint": "demand_constraint", "line": 42},
    "iis_expected": ["demand_constraint", "capacity_constraint"],
    "ground_truth_fix": "Change >= to <="
}
```

---

## Dataset Statistics Target

| Metric | Target |
|--------|--------|
| Total instances | 10,000+ |
| Error Type A | 2,500 (25%) |
| Error Type B | 2,500 (25%) |
| Error Type C | 2,500 (25%) |
| Error Type D | 2,500 (25%) |
| Avg IIS size | 3-5 constraints |
| Problem domains | 10+ (transportation, scheduling, etc.) |

---

## Quality Assurance

1. **Solvability Check**: Original models must solve to optimality
2. **Error Confirmation**: Sabotaged models must fail with target status
3. **Fix Verification**: Ground truth fix must restore feasibility
4. **Difficulty Calibration**: Balance easy (single constraint) vs hard (cascading errors)

---

*Next: [A4_Evaluation_Metrics.md](A4_Evaluation_Metrics.md) - Metrics and Supply Chain Extension*
