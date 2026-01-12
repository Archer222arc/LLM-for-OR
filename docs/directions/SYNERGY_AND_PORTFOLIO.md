# Strategic Portfolio: Synergies and Recommendations

## 5.5 Portfolio Approach

Rather than pursuing a single direction, we recommend a **portfolio strategy** that maximizes synergies:

---

## Publication Strategy

### Primary Submission (NeurIPS 2026)

**Direction A (OR-Debug-Bench)** as the flagship contribution.

| Criterion | Assessment |
|-----------|------------|
| Novelty | Clearest—no existing process-level debugging benchmark |
| MDP Formulation | Strongest—solver provides ideal environment |
| PILOT-Bench Continuity | Direct extension to mathematical optimization |
| Implementation Feasibility | Highest—Gurobi/Pyomo well-documented |

### Secondary Submission (NeurIPS 2026 or ICML 2027)

**Direction B (OR-Bias-Bench)** OR **Direction C (OR-Compliance-Bench)**

| Factor | Direction B | Direction C |
|--------|-------------|-------------|
| Data Availability | Moderate (simulation-based) | Lower (needs expert data) |
| Domain Expertise | Lower (classical OR theory) | Higher (trade compliance) |
| Practical Impact | High (inventory management) | High (trade automation) |

**Recommendation:** Choose based on data availability and advisor preference.

### Future Work (Job Talk)

**Directions D-H** as the "research agenda" demonstrating breadth and long-term vision.

---

## Synergy Map: How Directions Connect

| Direction Pair | Synergy | Combined Value Proposition |
|----------------|---------|----------------------------|
| **A + B** | Both evaluate "rationality" of LLM decisions | "Can LLMs make correct AND rational OR decisions?" |
| **A + C** | Both require multi-step reasoning with verification | "Sequential reasoning in constrained domains" |
| **A + D** | Debug = fix bad formulations; D = prevent them | "Full lifecycle of formulation quality" |
| **B + F** | Bias detection + causal reasoning for decisions | "Trustworthy agentic supply chain management" |
| **C + F** | Rule application + disruption response | "Adaptive compliance under uncertainty" |

---

## Job Talk Narrative Arc (Enhanced)

### Chapter 1: Foundation
**PILOT-Bench** established the foundation for evaluating LLM agents under uncertainty, introducing MDP perspective to workflow execution.

### Chapter 2: Verification
**OR-Debug-Bench** extends to mathematical optimization debugging, leveraging solver-backed verification to overcome the Self-Correction Blind Spot.

### Chapter 3: Rationality & Compliance
**OR-Bias-Bench / OR-Compliance-Bench** expand to behavioral rationality and regulatory compliance, demonstrating breadth across OR/OM.

### Chapter 4: Future Vision
Future directions in disruption reasoning, sim-to-real transfer, and safety-constrained RL complete the vision of **trustworthy agentic operations**.

---

## Research Timeline Visualization

```
2026 Q1-Q2: OR-Debug-Bench Development
    ├── Data: Saboteur Agent + MIPLIB
    ├── Environment: Gym wrapper for Gurobi
    └── Evaluation: Frontier LLM baselines

2026 Q2-Q3: OR-Debug-Bench Paper
    ├── NeurIPS 2026 submission
    └── Public benchmark release

2026 Q3-Q4: Secondary Direction
    ├── OR-Bias-Bench OR OR-Compliance-Bench
    └── ICML 2027 preparation

2027+: Expanded Portfolio
    ├── Directions D-F development
    └── Industry partnerships
```

---

## The Vision Statement

> Building AI systems that don't just translate requirements into code, but **iterate, debug, reason about biases, navigate regulations, and adapt to disruptions**—like expert practitioners.

> *This is the path from AI as a tool to AI as a trusted operational collaborator.*

---

## Immediate Next Steps

| Priority | Action | Timeline |
|----------|--------|----------|
| **1** | Finalize OR-Debug-Bench MDP specification with Saboteur Agent error taxonomy | Week 1-2 |
| **2** | Prototype data generation using OptMATH + MIPLIB with controlled error injection | Week 2-4 |
| **3** | Scope OR-Bias-Bench: Identify 3-5 classic OM decision problems with known optimal policies | Week 3-4 |
| **4** | Evaluate HSCodeComp feasibility for Direction C | Week 4-5 |
| **5** | Baseline evaluation on DeepSeek-R1, GPT-4o, Claude-3.5 | Week 5-8 |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data generation too slow | Medium | High | Start with smaller MIPLIB subset |
| Gurobi licensing issues | Low | Medium | Use open-source HiGHS as backup |
| Frontier LLMs too strong | Low | Medium | Design harder problems |
| Benchmark not discriminative | Medium | High | Iterative difficulty calibration |

---

*End of Research Directions Documentation*

*Original document archived at: `docs/archive/NeurIPS2026_Research_Directions_Report_ORIGINAL.md`*
