# TIER 1 Research Directions Overview

## Selection Criteria

> **TIER 1 Criteria:** Exceptional novelty (no existing benchmark), strong MDP formulation, irreplaceable LLM capabilities, clear literature gap, and high practical impact.

---

## Comprehensive Direction Comparison Matrix

| Direction | Specificity | MDP Novelty | LLM Irreplaceable | Literature Gap | Overall |
|-----------|-------------|-------------|-------------------|----------------|---------|
| **A: Solver Infeasibility Diagnosis** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | CorrectBench no OR | **TIER 1** |
| **B: Behavioral Bias Detection (OM)** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | AIM-Bench is new | **TIER 1** |
| **C: Hierarchical Rule Composition** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | HSCodeComp 46.8% | **TIER 1** |

---

## Direction A: Solver Infeasibility Diagnosis & Recovery (Primary)

**Benchmark Name:** OR-Debug-Bench

**Core Concept:** MDP-based debugging benchmark where LLMs diagnose IIS logs and generate code fixes.

**Why It Wins:**
- Solver returns incontrovertible feedback (IIS)
- GNNs cannot process semantic error messages
- CorrectBench (64.5% blind spot) omits OR entirely

**Key Innovation:** Process evaluation vs. outcome evaluation

**Detailed Documentation:** [A_OR_Debug_Bench/](A_OR_Debug_Bench/)

---

## Direction B: Behavioral Bias Detection in Agentic Decision-Making

**Benchmark Name:** OR-Bias-Bench

**Core Concept:** Evaluate and mitigate cognitive biases in LLM agents making operational decisions.

**Why It's Novel:**
- AIM-Bench (2025) first documented behavioral biases in OR agents
- LLMs inherit human-like biases from training data; GNNs don't
- Simply training to "maximize profit" is insufficient

**Key Findings from AIM-Bench:**
- "Pull-to-center" effect: under-ordering high-profit items
- Systematic deviation from rational Newsvendor solutions

**Detailed Documentation:** [B_OR_Bias_Bench.md](B_OR_Bias_Bench.md)

---

## Direction C: Hierarchical Rule Composition for Trade Compliance

**Benchmark Name:** OR-Compliance-Bench

**Core Concept:** Evaluate LLM agents' ability to apply complex, hierarchical rule systems in supply chain compliance.

**Why It's Novel:**
- HSCodeComp (ICLR 2026): 46.8% agent vs 95% human accuracy
- Trade compliance requires understanding legal text + multi-level exceptions
- Automation of compliance is high-value industry need

**Key Challenge:** Compositional generalization across rule hierarchies

**Detailed Documentation:** [C_OR_Compliance_Bench.md](C_OR_Compliance_Bench.md)

---

## Strategic Recommendation

> **Primary Submission (NeurIPS 2026):** Direction A (Solver Infeasibility Diagnosis) as the flagship contribution. This has the clearest novelty, strongest MDP formulation, and most direct PILOT-Bench continuity.

> **Secondary Submission (NeurIPS 2026 or ICML 2027):** Direction B (Behavioral Bias Detection) OR Direction C (Hierarchical Rule Composition). Both are highly novel with strong practical impact. Choose based on data availability and advisor preference.

---

## TIER 1 Common Characteristics

| Characteristic | Direction A | Direction B | Direction C |
|----------------|-------------|-------------|-------------|
| Ground-truth verification | Solver IIS | Known optimal policy | Regulatory acceptance |
| LLM-specific capability | NL error parsing | Bias detection in language | Legal text comprehension |
| MDP formulation | Debugging trajectory | Decision sequence | Rule application chain |
| Industry relevance | Solver debugging | Inventory management | Trade compliance |

---

*See also: [TIER2_Directions_DEF.md](TIER2_Directions_DEF.md) and [TIER3_Directions_GH.md](TIER3_Directions_GH.md)*
