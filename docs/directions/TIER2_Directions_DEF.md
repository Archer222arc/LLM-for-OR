# TIER 2 Research Directions: D, E, F

> **TIER 2 Criteria:** Strong novelty with some existing work to build on, good MDP fit, clear differentiation from prior art.

---

## Direction D: Formulation Quality & Efficiency Evaluation

### Core Concept

Move beyond binary correctness to evaluate the **quality, efficiency, and scalability** of mathematical formulations. Two correct models can differ by orders of magnitude in solve time due to formulation tightness.

### Literature Foundation

| Work | Contribution | Gap |
|------|--------------|-----|
| StepORLM | Evaluates step quality but focuses on reasoning, not solver performance | No formulation quality metrics |
| OptiTree | Demonstrates hierarchical decomposition yields better results | Formulation structure matters |
| OptiMUS | Notes solve time dependency on formulation | No systematic evaluation |

### Proposed Metrics

| Metric | Definition |
|--------|------------|
| **LP Relaxation Gap** | \|Z_IP - Z_LP\| / Z_IP (tightness of continuous relaxation) |
| **Symmetry Score** | Detection of interchangeable variables that confuse B&B |
| **Solve Time Ratio** | Agent formulation vs. expert formulation on same problem |
| **Constraint Efficiency** | Redundant constraint detection rate |

### MDP Formulation

- **State**: Current formulation + solver performance metrics (node count, LP relaxation value)
- **Action**: Add symmetry-breaking constraint, tighten Big-M, reformulate objective, add cuts
- **Reward**: Solve time reduction + optimality gap improvement

---

## Direction E: Sim-to-Real Policy Transfer for OR

### Core Concept

Evaluate robustness of OR policies when transferred from simulation to deployment. Sim2Act (ICLR 2026) addresses this for supply chains via adversarial calibration, but no benchmark systematically tests transfer quality.

### Why This Matters

| Point | Evidence |
|-------|----------|
| Industrial Validation | DeepStock deployed at Alibaba (1M+ SKUs) proves industrial scale is achievable |
| Key Barrier | "Reality gap" between simulator and deployment remains key barrier |
| Evaluation Gap | Need standardized evaluation of how policies degrade under distribution shift |

### Proposed Benchmark: "OR-Transfer-Bench"

| Phase | Description |
|-------|-------------|
| **Phase 1 (Training)** | Train policies on "clean" simulator with known dynamics |
| **Phase 2 (Transfer)** | Evaluate on perturbed environments (demand noise, lead time variability, capacity shocks) |

### Metrics

| Metric | Definition |
|--------|------------|
| **Performance Degradation Curve** | How performance drops with increasing perturbation |
| **Worst-Case Regret** | Maximum loss under adversarial perturbation |
| **Recovery Speed** | Time to adapt after distribution shift |

---

## Direction F: Causal Reasoning for Supply Chain Disruption

### Core Concept

Evaluate LLM agents' ability to reason about **causal propagation** of disruptions through supply networks. Causal-GNN SupplyNets (ICLR 2026) uses GNN-based causal world models, but LLMs offer unique advantages in interpreting unstructured disruption signals.

### LLM Advantage

| Capability | Why LLMs Excel |
|------------|----------------|
| Disruption Signals | News articles, supplier communications, regulatory announcements are unstructured text |
| Causal Chains | Understanding "Suez Canal blocked" → "semiconductor lead times increase" requires world knowledge |
| Current Practice | Human analysts combine news + network structure + domain knowledge; LLMs can approximate this |

### MDP Formulation

- **State**: Network topology, current disruption signals (text), inventory positions, demand forecasts
- **Action**: Expedite orders, activate backup suppliers, adjust safety stock, reroute shipments
- **Reward**: Service level maintained + cost of intervention + disruption recovery time

### Proposed Benchmark: "OR-Disruption-Bench"

| Scenario | Causal Chain |
|----------|--------------|
| **Semiconductor** | Taiwan earthquake → fab shutdown → automotive shortage |
| **Shipping** | Port congestion → container shortage → retail stockouts |
| **Energy** | Pipeline attack → fuel shortage → logistics cost spike |

### Metrics

| Metric | Definition |
|--------|------------|
| **Disruption Anticipation Accuracy** | Early warning capability |
| **Causal Chain Identification** | Correctly trace propagation path |
| **Mitigation Effectiveness** | Recovery speed and cost |

---

## TIER 2 Summary Table

| Direction | Novelty | Feasibility | Data Requirement |
|-----------|---------|-------------|------------------|
| D: Formulation Quality | Medium | High | MIPLIB + expert annotations |
| E: Sim-to-Real | Medium | Medium | Simulator development needed |
| F: Disruption Reasoning | High | Medium | News + supply chain data |

---

*See also: [TIER3_Directions_GH.md](TIER3_Directions_GH.md)*
