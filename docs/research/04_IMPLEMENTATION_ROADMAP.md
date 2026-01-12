# Implementation Roadmap to NeurIPS 2026

## 6.1 Proposed Paper Structure

**Title:** "OR-Debug-Bench: A Gym Environment for Solver Infeasibility Diagnosis and Recovery"

1. **Introduction**: Motivation (solver debugging as practitioner pain); gap (no process evaluation); contribution preview
2. **Related Work**: LLM-OR benchmarks (outcome-focused), self-correction (CorrectBench gap), process rewards (StepORLM)
3. **Benchmark Design**: MDP formulation, Saboteur Agent data generation, hierarchical action space
4. **Evaluation Framework**: Recovery Rate, Diagnosis Accuracy, Optimality Preservation, Trajectory Efficiency
5. **Experiments**: Frontier LLM evaluation (GPT-4o, Claude-3.5, DeepSeek-R1); GRPO training on LLaMA-3-8B
6. **Conclusion**: Findings, implications for deployment, Supply Chain Resilience extension

---

## 6.2 Timeline

| Phase | Timeline | Activities | Deliverable |
|-------|----------|------------|-------------|
| 1: Data Construction | Jan-Feb 2026 | OptMATH pipeline; Saboteur Agent; 10k+ instances with ground-truth fixes | OR-Debug-Dataset-v1 |
| 2: Environment Dev | Feb-Mar 2026 | Gym interface wrapping Gurobi/Pyomo; IIS parsers; reward calculators | `pip install or-debug-bench` |
| 3: Baselines & Training | Mar-Apr 2026 | Evaluate DeepSeek-R1, GPT-4o, Claude-3.5 zero-shot; Train LLaMA-3-8B with GRPO | Results tables + learning curves |
| 4: Paper Writing | Apr-May 2026 | Draft with "Process Evaluation" narrative; position against CorrectBench, StepORLM | NeurIPS 2026 submission |

---

## 6.3 Resource Requirements

- **Compute**:
  - Environment: Minimal (Gurobi runs on CPU)
  - GRPO Training: ~8×A100 for full training; 2×A100 sufficient for evaluation and LoRA fine-tuning
- **Software**: Gurobi Academic License (free), Python-Gurobi, PILOT-Bench codebase as foundation
- **Data**: Public sources: MIPLIB, NL4Opt, OptMATH, IndustryOR; synthetic generation via Saboteur Agent
- **API Costs**: Frontier LLM evaluation via API (GPT-4, Claude-3, Gemini, DeepSeek-R1)

---

## 6.4 Alignment with NeurIPS/ICLR Trends

| Trend | Relevant Papers | Connection to OR-Debug-Bench |
|-------|-----------------|------------------------------|
| RL with Verifiable Rewards | DeepSeek-R1, OR-R1, GRPO | Solver status is ultimate verifiable reward |
| Process Reward Models | StepORLM, BiPRM, PAVs | Diagnosis Accuracy metric rewards reasoning steps |
| Agentic Supply Chain | Causal-GNN, DeepStock, C.H. Robinson | Supply Chain Resilience track extends scope to OM |
| Self-Correction Research | CorrectBench, MOID | Directly addresses 64.5% blind spot in OR context |
| Test-Time Compute | OpenAI o3, OptiTree | Metrics include Trajectory Efficiency for thinking cost |

---

## Conclusion: A Research Portfolio for Agentic OR/OM

The era of "translation-based" benchmarks is over. The future lies in evaluating the **process of reasoning**, the **rationality of decisions**, and the **resilience of agents** across the full spectrum of Operations Research and Operations Management.

This report identifies **8 novel research directions** organized into three tiers, with three TIER 1 directions offering exceptional novelty and impact:

- **Direction A (Primary):** Solver Infeasibility Diagnosis & Recovery — addresses the "Self-Correction Blind Spot" with solver-backed verification
- **Direction B (High Priority):** Behavioral Bias Detection — evaluates rationality of LLM decisions, extending AIM-Bench insights across OM domains
- **Direction C (High Priority):** Hierarchical Rule Composition — tackles the 46.8% vs 95% gap in HSCodeComp with process-level evaluation for compliance

> **Strategic Portfolio Recommendation:** Submit Direction A (OR-Debug-Bench) to NeurIPS 2026 as the flagship contribution with strongest novelty and PILOT-Bench continuity. Develop Direction B or C as a secondary project for ICML 2027 or parallel submission. Use Directions D-H as the "future work" section of your job talk, demonstrating research breadth and long-term vision.

The emergence of reasoning models like OpenAI o3 and DeepSeek-R1 suggests that future OR/OM systems will not be static code generators but dynamic "thinking partners." They will diagnose errors, detect their own biases, navigate complex rule hierarchies, and adapt to disruptions. The benchmark portfolio proposed here provides the necessary gymnasium to train and evaluate these System 2 capabilities.

> **The Vision:** Building AI systems that don't just translate requirements into code, but iterate, debug, reason about biases, navigate regulations, and adapt to disruptions—like expert practitioners. *This is the path from AI as a tool to AI as a trusted operational collaborator.*

---

## Job Talk Narrative Arc

1. **Chapter 1 (Foundation):** PILOT-Bench established the foundation for evaluating LLM agents under uncertainty, introducing MDP perspective to workflow execution.
2. **Chapter 2 (Verification):** OR-Debug-Bench extends to mathematical optimization debugging, leveraging solver-backed verification to overcome the Self-Correction Blind Spot.
3. **Chapter 3 (Rationality & Compliance):** OR-Bias-Bench / OR-Compliance-Bench expand to behavioral rationality and regulatory compliance, demonstrating breadth across OR/OM.
4. **Chapter 4 (Future Vision):** Future directions in disruption reasoning, sim-to-real transfer, and safety-constrained RL complete the vision of trustworthy agentic operations.

---

## Immediate Next Steps

1. **Direction A:** Finalize OR-Debug-Bench MDP specification with Saboteur Agent error taxonomy
2. **Direction B:** Scope OR-Bias-Bench: Identify 3-5 classic OM decision problems with known optimal policies for bias evaluation
3. **Direction C:** Evaluate HSCodeComp feasibility: Assess data availability and regulatory expertise needed for rule composition benchmark
4. **Infrastructure:** Prototype data generation infrastructure using OptMATH + MIPLIB with controlled error injection
5. **Validation:** Baseline evaluation on DeepSeek-R1, GPT-4o, Claude-3.5 to validate benchmark discriminability across directions

---

*End of Implementation Roadmap*
