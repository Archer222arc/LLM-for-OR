# Direction A: OR-Debug-Bench — Evaluation Metrics

## Primary Metrics

| Metric | Definition | What It Measures |
|--------|------------|------------------|
| **Recovery Rate (RR)** | % of infeasible instances converted to feasible within T steps | Overall debugging capability |
| **Diagnosis Accuracy (DA)** | Correct identification of root cause (compare to ground-truth IIS) | Reasoning quality |
| **Optimality Preservation (OP)** | Deviation of recovered objective from original intended objective | Penalizes trivial "delete all" fixes |
| **Trajectory Efficiency (TE)** | Number of solver interactions to reach solution | Thinking cost; aligns with StepORLM |
| **Feasibility Preservation (FP)** | % of original correct constraints preserved during fix | Quality of repair strategy |

---

## Metric Definitions

### Recovery Rate (RR)

```
RR = (# instances recovered to OPTIMAL) / (# total infeasible instances)

Reported at multiple step budgets:
- RR@5: Recovery within 5 steps
- RR@10: Recovery within 10 steps
- RR@20: Recovery within 20 steps
```

### Diagnosis Accuracy (DA)

```
DA = |IIS_predicted ∩ IIS_actual| / |IIS_actual|

Where:
- IIS_predicted: Constraints agent identifies as problematic
- IIS_actual: Ground-truth IIS from solver
```

### Optimality Preservation (OP)

```
OP = 1 - |Z_recovered - Z_original| / |Z_original|

Where:
- Z_recovered: Objective value after fix
- Z_original: Objective value of original (pre-sabotage) model
```

### Trajectory Efficiency (TE)

```
TE = 1 / (# solver calls to recovery)

Higher is better. Normalized to [0, 1] range.
```

### Feasibility Preservation (FP)

```
FP = |C_preserved ∩ C_original| / |C_original|

Where:
- C_preserved: Constraints in recovered model
- C_original: Constraints in original model
```

---

## Leaderboard Format

| Model | RR@5 | RR@10 | DA | OP | TE | FP |
|-------|------|-------|----|----|----|----|
| GPT-4o | - | - | - | - | - | - |
| Claude-3.5 | - | - | - | - | - | - |
| DeepSeek-R1 | - | - | - | - | - | - |
| LLaMA-3-8B (GRPO) | - | - | - | - | - | - |

---

## Supply Chain Resilience Extension

To satisfy the expanded OM scope, OR-Debug-Bench includes a specialized **"Supply Chain Resilience"** track:

### Scenario Definition

Instead of "infeasible code," agent faces **"infeasible plan"** (demand exceeds capacity due to disruption).

### Task Description

| Component | Specification |
|-----------|---------------|
| **State** | Current network state, disruption signal, inventory levels |
| **Action** | Adjust inventory parameters, reroute shipments, relax SLAs |
| **Goal** | Restore operational feasibility within constraints |

### Connection to Core Benchmark

| OR-Debug-Bench Core | Supply Chain Extension |
|---------------------|------------------------|
| Solver infeasibility | Capacity infeasibility |
| IIS analysis | Bottleneck identification |
| Constraint relaxation | SLA relaxation |
| Code modification | Plan modification |

### Metrics Extension

| Metric | SC Interpretation |
|--------|-------------------|
| Recovery Rate | % plans restored to feasible |
| Diagnosis Accuracy | Correct bottleneck identification |
| Optimality Preservation | Service level maintenance |
| Trajectory Efficiency | Speed of recovery |

---

## Evaluation Protocol

1. **Zero-shot Evaluation**: Test frontier LLMs without fine-tuning
2. **Few-shot Evaluation**: Provide 3-5 examples of successful debugging
3. **Fine-tuned Evaluation**: GRPO-trained agents on OR-Debug-Bench
4. **Ablation Studies**:
   - Impact of IIS availability
   - Impact of error type distribution
   - Impact of problem complexity

---

*Back to: [TIER1_Overview.md](../TIER1_Overview.md) - Overview of All TIER 1 Directions*
