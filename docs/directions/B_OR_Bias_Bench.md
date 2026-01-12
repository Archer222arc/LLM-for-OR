# Direction B: OR-Bias-Bench — Behavioral Bias Detection

## Core Concept

Evaluate and mitigate cognitive biases in LLM agents making operational decisions. AIM-Bench (2025) revealed that LLM inventory managers exhibit systematic biases—the "pull-to-center" effect where agents under-order high-profit items and over-order low-profit ones, deviating from rational Newsvendor solutions.

---

## Why This is Novel

| Aspect | Justification |
|--------|---------------|
| **Literature Gap** | AIM-Bench is the first to document behavioral biases in OR agents, but only covers inventory |
| **LLM-Specific** | LLMs inherit human-like cognitive biases from training data; GNNs/heuristics don't have this problem |
| **Practical Impact** | Simply training to "maximize profit" is insufficient—must explicitly evaluate rationality |

---

## MDP Formulation

### State Space
- Market conditions (demand distribution parameters)
- Demand signals (historical data, forecasts)
- Inventory levels (current stock, pipeline)
- Historical decisions (past order quantities)

### Action Space
- Order quantities
- Pricing decisions
- Capacity allocations

### Reward Function
```
R = Profit + Rationality_Penalty

Where:
Rationality_Penalty = -λ * |decision - optimal_policy(state)|

optimal_policy is computed from known distribution (Newsvendor, etc.)
```

---

## Proposed Benchmark: "OR-Bias-Bench"

### Scenarios

| Scenario | Classical Problem | Optimal Policy |
|----------|-------------------|----------------|
| **Single-item Newsvendor** | Perishable inventory | Critical ratio formula |
| **Dynamic Pricing** | Revenue management | Weber-Talluri policy |
| **Multi-item Allocation** | Capacity allocation | LP relaxation |
| **Sequential Decision** | Multi-period inventory | Base-stock policy |

### Metrics

| Metric | Definition |
|--------|------------|
| **Bias Detection Rate** | Identify when agent deviates from rational baseline |
| **Bias Magnitude** | Quantify economic loss from biased decisions |
| **Debiasing Effectiveness** | Improvement after intervention (prompting, fine-tuning) |

---

## Key Biases to Evaluate

### From AIM-Bench

| Bias | Description | OR Context |
|------|-------------|------------|
| **Pull-to-center** | Order quantities regress toward mean demand | Under-ordering high-profit items |
| **Anchoring** | Over-reliance on initial information | Sticking to initial forecasts |
| **Loss aversion** | Overweighting potential losses | Over-ordering to avoid stockouts |
| **Recency bias** | Overweighting recent observations | Ignoring long-term trends |

### Additional Biases

| Bias | Description | OR Context |
|------|-------------|------------|
| **Framing effects** | Decision changes based on problem presentation | Profit vs. loss framing |
| **Status quo bias** | Preference for current state | Resistance to policy changes |
| **Overconfidence** | Underestimating uncertainty | Narrow safety stock margins |

---

## Extension to OM Domains

Beyond inventory, extend bias evaluation to:

1. **Hiring Decisions with Fairness Constraints**
   - Detect discrimination in resource allocation
   - Evaluate compliance with fairness metrics

2. **Resource Allocation Under Uncertainty**
   - Multi-objective trade-offs
   - Risk-averse vs. risk-neutral behavior

3. **Pareto-Front Selection**
   - Biases in multi-objective optimization
   - Preference for dominated solutions

---

## Data Generation Strategy

```python
class BiasEvaluationEnvironment:
    def __init__(self, problem_type, distribution_params):
        self.problem = problem_type  # "newsvendor", "pricing", etc.
        self.params = distribution_params
        self.optimal_policy = compute_optimal_policy(problem_type, params)

    def evaluate_bias(self, agent_decisions, states):
        optimal_decisions = [self.optimal_policy(s) for s in states]
        bias_magnitude = compute_deviation(agent_decisions, optimal_decisions)
        bias_type = classify_bias_pattern(agent_decisions, optimal_decisions)
        return {"magnitude": bias_magnitude, "type": bias_type}
```

---

## Connection to Direction A

| Direction A | Direction B | Synergy |
|-------------|-------------|---------|
| Evaluates correctness | Evaluates rationality | "Can LLMs make correct AND rational OR decisions?" |
| Solver verification | Policy comparison | Both use ground-truth verification |
| Debugging process | Decision process | Both evaluate multi-step reasoning |

---

*Back to: [TIER1_Overview.md](TIER1_Overview.md)*
