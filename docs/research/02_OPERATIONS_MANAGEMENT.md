# Operations Management Expansion — The Frontier of Agentic AI

Moving beyond mathematical solving mechanics, this document expands scope to Operations Management (OM). Literature from late 2025 indicates rapid evolution from predictive analytics to autonomous, agentic workflows capable of managing supply chains, inventory, and service operations.

---

## 3.1 The Agentic Supply Chain: From Prediction to Resilience

The "Agentic Supply Chain" has transitioned from theoretical ideal to industrial reality. C.H. Robinson launched agentic platforms that autonomously execute decisions—negotiating rates, rerouting shipments, managing exceptions without human intervention. This represents a shift from "human-in-the-loop" to "human-on-the-loop" or "human-out-of-the-loop."

### 3.1.1 Causal-GNN SupplyNets and Resilience

The Causal-GNN SupplyNets framework (ICLR 2026) addresses supply chain resilience by unifying causal reasoning with robust control. It employs a Causal World Model based on GNNs to predict how micro-level disruptions propagate through macro-level networks. Crucially, it uses a Lyapunov-safe RL controller ensuring critical operational constraints are satisfied with high probability.

### 3.1.2 Sim2Act: Bridging the Reality Gap

Sim2Act tackles the "reality gap" between simulation and deployment through adversarial calibration (reduce simulator bias) and group-relative perturbations (robustify policy). This ensures agents trained in digital twins perform reliably in noisy real-world environments.

---

## 3.2 Inventory Management: Biases and Regularization

### 3.2.1 DeepStock: Industrial-Scale Deployment

DeepStock (NeurIPS 2025 MLxOR workshop) represents a breakthrough in deployable RL for inventory control. It incorporates policy regularizations from classical inventory theory (e.g., base-stock-like behavior constraints). This hybrid neuro-symbolic approach enabled full-scale deployment on Alibaba's Tmall, managing over 1 million SKUs.

### 3.2.2 AIM-Bench: Exposing Behavioral Biases

AIM-Bench exposes LLM agent vulnerabilities. In simulated Newsvendor problems, agents demonstrated significant behavioral biases—the "pull-to-center" effect (under-ordering high-profit items, over-ordering low-profit ones). This finding is critical: simply training to "maximize profit" is insufficient. Benchmarks must explicitly test for and penalize irrational cognitive biases.

---

## 3.3 Service Operations: Dynamic Coordination

The "Create-to-Reuse" framework (ICLR 2026) models decisions on whether to reuse existing solutions or generate new ones at cost. In service contexts, this balances efficiency of standardized protocols against precision of customized care. Multi-agent systems for hospital operations use specialized agents (bed availability, staff schedules, patient inflow) that negotiate to optimize throughput.

---

## Summary Table: OM Literature and Implications

| Work | Domain | Key Finding | Implication for OR-Debug-Bench |
|------|--------|-------------|--------------------------------|
| C.H. Robinson Agentic | Supply Chain | Industrial deployment of autonomous agents | Validates market demand for agentic OR |
| Causal-GNN SupplyNets | Resilience | Lyapunov-safe RL for constraints | Safety-constrained RL is essential |
| Sim2Act | Sim-to-Real | Adversarial calibration reduces reality gap | Robustness testing required |
| DeepStock | Inventory | 1M+ SKUs at Alibaba Tmall | Neuro-symbolic approach works at scale |
| AIM-Bench | Inventory | "Pull-to-center" bias in LLM agents | Must evaluate behavioral rationality |
| HSCodeComp | Supply Chain Rules | 46.8% agent vs 95% human accuracy | Compositional generalization is hard |

---

## Connection to Research Directions

| OM Topic | Related Direction | Connection |
|----------|-------------------|------------|
| Supply Chain Resilience | Direction A Extension | OR-Debug-Bench includes "Supply Chain Resilience" track |
| Behavioral Biases | Direction B | OR-Bias-Bench evaluates rationality of decisions |
| Trade Compliance | Direction C | OR-Compliance-Bench for hierarchical rules |
| Sim-to-Real Transfer | Direction E | OR-Transfer-Bench for policy robustness |
| Disruption Reasoning | Direction F | OR-Disruption-Bench for causal propagation |

---

*Next: [03_LITERATURE_REVIEW.md](03_LITERATURE_REVIEW.md) - Complete 70+ Paper Reference*
