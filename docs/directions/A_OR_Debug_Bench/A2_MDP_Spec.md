# Direction A: OR-Debug-Bench — MDP Specification

## Environment: Solver as Gym

### State Space (Sₜ)

| Component | Description | Data Source |
|-----------|-------------|-------------|
| **Natural Language Problem** | Original problem description | MIPLIB, OptMATH, IndustryOR |
| **Current Code** | Python/Pyomo model code | Generated or sabotaged |
| **Solver Status** | Infeasible / Unbounded / Optimal / Error | Gurobi/CPLEX output |
| **IIS Log** | Set of conflicting constraints | `model.computeIIS()` |
| **History** | Trace of previous edit attempts | Trajectory buffer |

### Initial State Generation

Start with feasible instances from MIPLIB and IndustryOR. Use **"Saboteur Agent"** to inject synthetic errors (flip inequalities, tighten bounds to impossible levels) to create initial S₀.

---

### Action Space (Aₜ)

Hierarchical Tool Use design:

#### Diagnosis Actions
- `Get_IIS()` - Retrieve irreducible infeasible subsystem
- `Check_Slack()` - Examine constraint slack values
- `Explain_Conflict()` - Generate natural language explanation of conflict

#### Repair Actions
- `Relax_Constraint(id, ε)` - Relax constraint by epsilon
- `Drop_Constraint(id)` - Remove constraint entirely
- `Rewrite_Constraint(code)` - Provide new constraint code

#### Meta Actions
- `Restart` - Reset to initial state
- `Submit` - Submit current model for evaluation
- `Query_User` - Request clarification (optional)

Modeled after MOID workflow, allowing agents to query solver for diagnostic info before proposing fixes.

---

### Reward Function (Rₜ)

#### Outcome Rewards (Verifiable)
| Condition | Reward |
|-----------|--------|
| Feasible AND Gap < δ | +100 |
| Still Infeasible | -50 |
| Syntax Error | -50 |

#### Process Rewards (Dense)
| Condition | Reward |
|-----------|--------|
| IIS size reduced | +10 |
| Per step penalty | -1 |
| Constraint preserved | +5 |

#### Faithfulness Penalties
| Condition | Penalty |
|-----------|---------|
| Diagnosis contradicts solver log | -20 |
| Explanation inconsistent with action | -10 |

> **Key Design Principle:** Solver acts as ground-truth oracle, enabling RLVR paradigm. Process rewards address credit assignment for long debugging trajectories.

---

## Episode Structure

```
Episode Start:
    s₀ = (problem_nl, broken_code, INFEASIBLE, iis_log, [])

While not terminal:
    a_t = agent.act(s_t)
    s_{t+1}, r_t, done = env.step(a_t)

Terminal Conditions:
    - Solver returns OPTIMAL → Success
    - Max steps reached → Timeout
    - Agent submits broken code → Failure
```

---

## Implementation Notes

### Gurobi Integration
```python
import gurobipy as gp

def get_state(model):
    model.optimize()
    status = model.Status

    if status == gp.GRB.INFEASIBLE:
        model.computeIIS()
        iis = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
        return {"status": "INFEASIBLE", "iis": iis}
    elif status == gp.GRB.OPTIMAL:
        return {"status": "OPTIMAL", "objective": model.ObjVal}
    # ... handle other statuses
```

### Pyomo Integration
```python
from pyomo.environ import *

def get_state(model, solver):
    results = solver.solve(model)
    if results.solver.termination_condition == TerminationCondition.infeasible:
        # Use Gurobi's IIS through direct interface
        return {"status": "INFEASIBLE", "iis": extract_iis(model)}
```

---

*Next: [A3_Data_Generation.md](A3_Data_Generation.md) - Saboteur Agent and Error Injection*
