# NeurIPS 2026 Research Directions Overview

## The Agentic Turn in Operations Research
### From Static Translation to Dynamic Self-Correction

**Extending PILOT-Bench with MDP-Based Evaluation Frameworks**

*Comprehensive Literature Review: 70+ Papers (2024-2026)*

*Prepared for: Ruicheng Ao, MIT IDSS | January 2026*

---

## Executive Summary

This report outlines a strategic research agenda for NeurIPS 2026, synthesizing the rapidly evolving landscape of Large Language Model (LLM) agents within Operations Research (OR) and Operations Management (OM). The analysis is grounded in a comprehensive review of 70+ papers from 2024 through January 2026, identifying a fundamental paradigm shift from static, one-shot optimization tasks to dynamic, sequential, and "agentic" decision-making processes.

> **Central Thesis:** The current evaluation infrastructure for AI in OR is insufficient. Existing benchmarks, while valuable for assessing formulation accuracy, fail to capture the iterative nature of solver debugging, the stochastic complexity of supply chain resilience, and the behavioral nuances of agentic inventory management.

Our analysis of NeurIPS 2025 and ICLR 2026 proceedings reveals three converging trends that validate this direction:

- **Reinforcement Learning with Verifiable Rewards (RLVR)**: As demonstrated by DeepSeek-R1 and OR-R1, achieving 6%+ improvement in OR tasks
- **Process Reward Models (PRMs)**: StepORLM and BiPRM showing dense feedback outperforms sparse outcome signals
- **Agentic Workflows in Production**: C.H. Robinson's industrial deployment; DeepStock managing 1M+ SKUs at Alibaba Tmall

> **Primary Recommendation:** Direction A: Self-Correction Evaluation Benchmark, instantiated as **"OR-Debug-Bench: Solver Infeasibility Diagnosis and Recovery."** This direction addresses the "Self-Correction Blind Spot" (64.5% failure rate in existing benchmarks) by leveraging solver-backed ground-truth verification—making LLM's role unchallengeable vs. GNNs.

---

## Document Structure

This research report is organized into the following documents:

| Document | Content |
|----------|---------|
| `00_PROJECT_OVERVIEW.md` | This file - Executive summary and navigation |
| `01_RESEARCH_CONTEXT.md` | Crisis of static evaluation + Algorithmic renaissance |
| `02_OPERATIONS_MANAGEMENT.md` | Agentic supply chain, inventory, service operations |
| `03_LITERATURE_REVIEW.md` | Complete 70+ paper reference (7 categories) |
| `04_IMPLEMENTATION_ROADMAP.md` | Timeline, resources, paper structure |

### Research Directions

| Direction | Tier | Document |
|-----------|------|----------|
| **A: OR-Debug-Bench** | 1 | `directions/A_OR_Debug_Bench/` (4 files) |
| **B: OR-Bias-Bench** | 1 | `directions/B_OR_Bias_Bench.md` |
| **C: OR-Compliance-Bench** | 1 | `directions/C_OR_Compliance_Bench.md` |
| **D-F** | 2 | `directions/TIER2_Directions_DEF.md` |
| **G-H** | 3 | `directions/TIER3_Directions_GH.md` |
| **Portfolio Strategy** | - | `directions/SYNERGY_AND_PORTFOLIO.md` |

---

## Quick Reference: Research Direction Comparison

| Direction | Specificity | MDP Novelty | LLM Irreplaceable | Literature Gap | Overall |
|-----------|-------------|-------------|-------------------|----------------|---------|
| **A: Solver Infeasibility Diagnosis** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | CorrectBench no OR | **TIER 1** |
| **B: Behavioral Bias Detection (OM)** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | AIM-Bench is new | **TIER 1** |
| **C: Hierarchical Rule Composition** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | HSCodeComp 46.8% | **TIER 1** |
| D: Formulation Quality Evaluation | ⭐⭐ | ⭐⭐ | ⭐⭐ | StepORLM partial | TIER 2 |
| E: Sim-to-Real Policy Transfer | ⭐⭐ | ⭐⭐ | ⭐⭐ | Sim2Act exists | TIER 2 |
| F: Causal Disruption Reasoning | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Causal-GNN new | TIER 2 |
| G: Safety-Constrained RL for Ops | ⭐⭐ | ⭐⭐ | ⭐ | Lyapunov-safe exists | TIER 3 |
| H: Multi-Agent Service Coordination | ⭐⭐ | ⭐ | ⭐⭐ | Hospital AI exists | TIER 3 |

---

*For the complete original document, see: `docs/archive/NeurIPS2026_Research_Directions_Report_ORIGINAL.md`*
