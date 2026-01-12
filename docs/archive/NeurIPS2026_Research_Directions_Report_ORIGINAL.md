# NeurIPS 2026 Research Directions Report
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

## Part 1: The Crisis of Static Evaluation in Operations Research

### 1.1 The Limitations of the "Translation" Paradigm

For the past several years, the intersection of AI and OR has been dominated by a singular metaphor: translation. Researchers have treated optimization modeling primarily as a linguistic challenge—translating natural language problem descriptions into valid mathematical code. This paradigm has given rise to benchmarks like NL4Opt, OptiBench, and IndustryOR that evaluate one-shot formulation accuracy.

> **The Validity Void:** Real-world OR is rarely a linear process of translating text to code. It is a cyclical, iterative, diagnostic practice: construct models → attempt to solve → encounter infeasibility → analyze IIS → refine constraints. Agents that perform well on static leaderboards fail catastrophically in stochastic, error-prone industrial environments.

#### Table 1: Evolution of OR Benchmarks

| Benchmark | Year | Focus | Critical Limitation |
|-----------|------|-------|---------------------|
| NL4Opt | 2022 | NL → LP Translation | Static; no solver interaction |
| OptiBench | 2024 | Formulation Accuracy | One-shot; ignores infeasibility diagnosis |
| IndustryOR | 2024 | Industrial Relevance | Hard problems, but still static evaluation |
| OptMATH | 2025 | Data Synthesis Scale | Focuses on data, not agentic interaction |
| ORQA | 2024 | Domain Knowledge QA | Tests textbook knowledge, not practical solving |
| PILOT-Bench | 2026 | Workflow Robustness | Foundation for MDPs; not math debugging focus |
| **OR-Debug-Bench** | **2026 (Proposed)** | **Diagnosis & Recovery** | **Dynamic MDP; tests self-correction & reasoning** |

### 1.2 The "Gym Gap": Missing Infrastructure for Sequential Decision-Making

A granular analysis of ICLR 2026 and NeurIPS 2025 submissions reveals a striking infrastructural deficit: there is no dedicated Gym-like MDP environment for Operations Research. The RL community has developed sophisticated environments for robotics (MuJoCo), game playing (StarCraft), and software engineering (SWE-bench), yet the optimization solver—perhaps the most rigorous "environment" available—remains underutilized as a training ground.

The distinction is crucial. In a Gym environment:
- **Action**: The submission of a model or modification to a constraint
- **State**: Solver status (Optimal, Infeasible, Unbounded), duality gap, branch-and-bound node count, error logs
- **Reward**: Objective value or penalty for infeasibility—deterministic, noise-free, verifiable

> **Key Insight:** By failing to formalize OR as an MDP, the community has restricted itself to supervised fine-tuning on static datasets, ignoring the massive potential of RL to train agents that can *navigate* the solution space rather than just predict it.

### 1.3 The "Self-Correction Blind Spot"

The "Self-Correction Blind Spot" is a phenomenon where models can fix external errors (flagged by humans) but fail to identify and correct their own internal logic errors. CorrectBench (2025) documents a 64.5% blind spot rate across general domains.

#### Table 2: The Self-Correction Blind Spot in OR Context

| Error Type | Traditional LLM Behavior | Agentic Goal (OR-Debug-Bench) |
|------------|--------------------------|-------------------------------|
| Syntax Error | Retry with random variation | Analyze error message; fix specific syntax |
| Semantic Infeasibility | Fail; output "No Solution" | Request IIS; identify conflict; relax weakest constraint |
| Unboundedness | Hallucinate bounds | Diagnose missing constraint; query user for bounds |
| Biased Decision | "Pull-to-center" effect (AIM-Bench) | Detect bias via simulation; adjust policy parameters |

> **The OR Advantage:** Unlike creative writing where "correctness" is subjective, an optimization solver returns precise, incontrovertible feedback. Gurobi generates an IIS isolating exact conflicting constraints. This transforms self-correction from "introspective guessing" to "diagnostic reasoning."

---

## Part 2: The Algorithmic Renaissance (NeurIPS 2025 & ICLR 2026)

### 2.1 Reinforcement Learning with Verifiable Rewards (RLVR)

#### 2.1.1 The DeepSeek-R1 Moment

DeepSeek-R1 (January 2025) demonstrated that reasoning capabilities—self-reflection, verification, dynamic strategy adaptation—can emerge from pure RL when rewards are verifiable. It utilized Group Relative Policy Optimization (GRPO), eliminating the critic network for 50% memory reduction.

The implications for OR are profound: optimization is by definition a domain of verifiable rewards. The solver provides deterministic, noise-free signals ideal for RLVR.

#### 2.1.2 OR-R1: Domain-Specific RLVR

OR-R1 applies Task-specific GRPO (TGRPO) to OR problems, achieving 6%+ improvement in solving accuracy and drastically reducing the gap between single-attempt and multi-attempt performance. This validates that RLVR principles transfer effectively to the optimization domain.

#### 2.1.3 Emergent "Aha Moments"

DeepSeek-R1's technical report documents "aha moments"—instances where the model, driven solely by RL objective, learns to pause, re-evaluate, and self-correct during generation. In OR, this mirrors the cognitive process of realizing mid-formulation that a constraint is too tight. Capturing these moments is a key objective.

| Work | Method | Key Achievement | Relevance to OR-Debug-Bench |
|------|--------|-----------------|------------------------------|
| DeepSeek-R1 | GRPO | Emergent reasoning without supervised CoT | Validates pure RL can induce self-correction |
| OR-R1 | TGRPO | 6%+ improvement on OR tasks | Domain-specific validation of RLVR |
| Tülu 3 | RLVR | +1.7 MATH, +3.3 GSM8K | Confirms verifiable rewards effectiveness |
| DeepSeekMath | GRPO | 51.7% on MATH with 7B model | Shows small models can achieve strong reasoning |

### 2.2 Process Reward Models (PRMs) and Generative Verification

While Outcome Reward Models (ORMs) provide sparse signals at trajectory end, Process Reward Models (PRMs) provide dense, step-by-step feedback. This is vital for complex OR problems where correct final answers might arise from flawed reasoning (false positive) or minor errors in correct derivations cause failure (false negative).

#### 2.2.1 StepORLM: State-of-the-Art for OR

StepORLM (December 2025) utilizes a Generative Process Reward Model (GenPRM) that co-evolves with the policy model. Instead of classifying steps as "correct/incorrect," the GenPRM generates critiques evaluating logical consistency. This dual-feedback mechanism outperforms vastly larger generalist models on IndustryOR and ComplexOR.

#### 2.2.2 Bidirectional Process Supervision

BiPRM advances PRM research by looking both backward at context and forward at implications. In OR, this is analogous to understanding that a variable defined in line 1 only becomes problematic when interacting with a constraint in line 50. Integrating bidirectional supervision is essential for evaluating long-horizon dependencies.

| Work | Innovation | Key Result | Source |
|------|------------|------------|--------|
| StepORLM | GenPRM with dual-feedback | Outperforms larger models on IndustryOR | OpenReview 2025 |
| BiPRM | Bidirectional context modeling | Better credit assignment for long trajectories | arXiv 2025 |
| Math-Shepherd | Monte Carlo process estimation | Automatic process supervision at scale | ACL 2024 |
| PAVs | Process Advantage Verifiers | 6× sample efficiency over ORMs | ICLR 2025 |
| ThinkPRM | Generative verification | 1% training data needed | OpenReview 2025 |
| TaTToo | Tool-grounded thinking PRM | Tabular reasoning improvement | arXiv 2025 |

### 2.3 Test-Time Compute and the "Thinking" Agent

OpenAI o3 (January 2026) has solidified "test-time compute" as a dominant scaling law. These models "think" before speaking, generating extensive internal reasoning traces before committing to answers. O3 achieved 71.7% on SWE-bench Verified.

For OR, this shift from "fast" to "slow" thinking is transformative. A standard LLM might rush to output code, often hallucinating constraints. A reasoning model can allocate compute to simulate constraint implications, check logical consistency, and plan structure before writing code.

> **Benchmark Implication:** OR-Debug-Bench must measure not just output accuracy, but the efficiency and quality of the "thinking" process itself—trajectory length, diagnostic accuracy, and reasoning faithfulness.

---

## Part 3: Operations Management Expansion — The Frontier of Agentic AI

Moving beyond mathematical solving mechanics, this report expands scope to Operations Management (OM). Literature from late 2025 indicates rapid evolution from predictive analytics to autonomous, agentic workflows capable of managing supply chains, inventory, and service operations.

### 3.1 The Agentic Supply Chain: From Prediction to Resilience

The "Agentic Supply Chain" has transitioned from theoretical ideal to industrial reality. C.H. Robinson launched agentic platforms that autonomously execute decisions—negotiating rates, rerouting shipments, managing exceptions without human intervention. This represents a shift from "human-in-the-loop" to "human-on-the-loop" or "human-out-of-the-loop."

#### 3.1.1 Causal-GNN SupplyNets and Resilience

The Causal-GNN SupplyNets framework (ICLR 2026) addresses supply chain resilience by unifying causal reasoning with robust control. It employs a Causal World Model based on GNNs to predict how micro-level disruptions propagate through macro-level networks. Crucially, it uses a Lyapunov-safe RL controller ensuring critical operational constraints are satisfied with high probability.

#### 3.1.2 Sim2Act: Bridging the Reality Gap

Sim2Act tackles the "reality gap" between simulation and deployment through adversarial calibration (reduce simulator bias) and group-relative perturbations (robustify policy). This ensures agents trained in digital twins perform reliably in noisy real-world environments.

### 3.2 Inventory Management: Biases and Regularization

#### 3.2.1 DeepStock: Industrial-Scale Deployment

DeepStock (NeurIPS 2025 MLxOR workshop) represents a breakthrough in deployable RL for inventory control. It incorporates policy regularizations from classical inventory theory (e.g., base-stock-like behavior constraints). This hybrid neuro-symbolic approach enabled full-scale deployment on Alibaba's Tmall, managing over 1 million SKUs.

#### 3.2.2 AIM-Bench: Exposing Behavioral Biases

AIM-Bench exposes LLM agent vulnerabilities. In simulated Newsvendor problems, agents demonstrated significant behavioral biases—the "pull-to-center" effect (under-ordering high-profit items, over-ordering low-profit ones). This finding is critical: simply training to "maximize profit" is insufficient. Benchmarks must explicitly test for and penalize irrational cognitive biases.

| Work | Domain | Key Finding | Implication for OR-Debug-Bench |
|------|--------|-------------|--------------------------------|
| C.H. Robinson Agentic | Supply Chain | Industrial deployment of autonomous agents | Validates market demand for agentic OR |
| Causal-GNN SupplyNets | Resilience | Lyapunov-safe RL for constraints | Safety-constrained RL is essential |
| Sim2Act | Sim-to-Real | Adversarial calibration reduces reality gap | Robustness testing required |
| DeepStock | Inventory | 1M+ SKUs at Alibaba Tmall | Neuro-symbolic approach works at scale |
| AIM-Bench | Inventory | "Pull-to-center" bias in LLM agents | Must evaluate behavioral rationality |
| HSCodeComp | Supply Chain Rules | 46.8% agent vs 95% human accuracy | Compositional generalization is hard |

### 3.3 Service Operations: Dynamic Coordination

The "Create-to-Reuse" framework (ICLR 2026) models decisions on whether to reuse existing solutions or generate new ones at cost. In service contexts, this balances efficiency of standardized protocols against precision of customized care. Multi-agent systems for hospital operations use specialized agents (bed availability, staff schedules, patient inflow) that negotiate to optimize throughput.

---

## Part 4: Primary Research Direction — OR-Debug-Bench

### 4.1 Why Self-Correction Evaluation Wins

#### Argument 1: Specificity (The "GNN Defense")

In OR-Debug-Bench, state includes natural language error logs (e.g., "Model is infeasible. IIS indicates conflict between Constraint A and Constraint B"). Interpreting these semantic logs, mapping them to original problems, and generating Python fixes fundamentally requires LLM capabilities. GNNs operate on fixed graph structures—they cannot process semantic error messages or rewrite executable code. LLM's role is unchallengeable.

#### Argument 2: MDP Novelty (Process vs. Outcome)

Existing benchmarks measure P(correct code | text). OR-Debug-Bench measures policy π(aₜ | sₜ) where agents navigate erroneous states to find feasible regions. This enables evaluation of efficiency (steps to fix), robustness (does fix break other constraints), and reasoning quality (diagnosis accuracy). Metrics shift from "Pass@1" to Recovery Rate, Trajectory Efficiency, and Feasibility Preservation.

#### Argument 3: Literature Validation

- **CorrectBench**: Documents 64.5% blind spot rate in self-correction—massive unaddressed vulnerability in OR where single constraint errors break systems
- **MOID**: Explicitly validates LLMs can perform infeasibility diagnosis, generating constraint-aware heuristics and Pareto fronts of "least-violating" solutions
- **StepORLM**: Provides blueprint for process supervision in OR; OR-Debug-Bench extends this to the debugging phase
- **Gap Identification**: CorrectBench omits constrained optimization entirely—creating "blue ocean" opportunity for OR-specific contribution

### 4.2 Formal MDP Specification

#### Environment: Solver as Gym

| Component | Specification | Data Source / Rationale |
|-----------|---------------|-------------------------|
| **State Space (Sₜ)** | • Natural Language Problem<br>• Current Code (Python/Pyomo)<br>• Solver Status: Infeasible/Unbounded/Optimal<br>• IIS Log: Set of conflicting constraints<br>• History: Trace of previous edit attempts | Start with feasible instances from MIPLIB and IndustryOR. Use "Saboteur Agent" to inject synthetic errors (flip inequalities, tighten bounds to impossible levels) to create initial S₀. |
| **Action Space (Aₜ)** | Hierarchical Tool Use:<br>• Diagnosis: Get_IIS(), Check_Slack(), Explain_Conflict()<br>• Repair: Relax_Constraint(id, ε), Drop_Constraint(id), Rewrite_Constraint(code)<br>• Meta: Restart, Submit | Modeled after MOID workflow, allowing agents to query solver for diagnostic info before proposing fixes. |
| **Reward Function (Rₜ)** | Verifiable (Outcome):<br>• +100 if Feasible AND Gap < δ<br>Process (Dense):<br>• +10 for reducing IIS size<br>• -1 per step (efficiency)<br>• -50 for SYNTAX_ERROR<br>Faithfulness:<br>• Penalty if diagnosis contradicts solver log | Solver acts as ground-truth oracle, enabling RLVR paradigm. Process rewards address credit assignment. |

### 4.3 Data Generation Strategy

Leverage OptMATH pipeline and public libraries for synthetic error injection:

1. **Seed Generation**: Utilize ~200,000 feasible instances from OptMATH or MIPLIB as "Ground Truth"
2. **Systematic Error Injection**: Develop "Saboteur Agent" injecting common errors:
   - Type A (Bound Error): Swapping ≤ for ≥ in constraints
   - Type B (Variable Error): Defining integer as continuous or vice versa
   - Type C (Logic Error): Omitting summation term or index in loop
   - Type D (Conflict Error): Adding directly contradicting constraints
3. **Validation**: Run Gurobi on injected code to confirm target error status (INFEASIBLE/UNBOUNDED)
4. **Dataset Creation**: Pair (Original NL Description, Broken Code) forms initial state s₀

### 4.4 Evaluation Metrics

| Metric | Definition | What It Measures |
|--------|------------|------------------|
| Recovery Rate (RR) | % of infeasible instances converted to feasible within T steps | Overall debugging capability |
| Diagnosis Accuracy (DA) | Correct identification of root cause (compare to ground-truth IIS) | Reasoning quality |
| Optimality Preservation (OP) | Deviation of recovered objective from original intended objective | Penalizes trivial "delete all" fixes |
| Trajectory Efficiency (TE) | Number of solver interactions to reach solution | Thinking cost; aligns with StepORLM |
| Feasibility Preservation (FP) | % of original correct constraints preserved during fix | Quality of repair strategy |

### 4.5 Supply Chain Resilience Extension

To satisfy the expanded OM scope, OR-Debug-Bench includes a specialized "Supply Chain Resilience" track:

- **Scenario**: Instead of "infeasible code," agent faces "infeasible plan" (demand exceeds capacity due to disruption)
- **Task**: Adjust inventory parameters, reroute shipments, or relax service level agreements to restore operational feasibility
- **Connection**: Links solver-level debugging to system-level resilience problems from Causal-GNN SupplyNets and DeepStock

---

## Part 5: Expanded Research Directions Beyond Debugging

Beyond the primary Self-Correction direction, the literature review reveals multiple high-potential research directions spanning Operations Research and Operations Management. We identify **8 novel directions** organized into three tiers based on novelty, feasibility, and alignment with emerging trends.

### 5.1 Comprehensive Direction Comparison Matrix

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

### 5.2 TIER 1: Highest-Priority Novel Directions

> **TIER 1 Criteria:** Exceptional novelty (no existing benchmark), strong MDP formulation, irreplaceable LLM capabilities, clear literature gap, and high practical impact.

#### Direction A: Solver Infeasibility Diagnosis & Recovery (Primary)

Covered in Part 4. Core contribution: MDP-based debugging benchmark where LLMs diagnose IIS logs and generate code fixes. Literature gap: CorrectBench omits constrained optimization; MOID validates feasibility but lacks standardized evaluation.

---

#### Direction B: Behavioral Bias Detection in Agentic Decision-Making ⭐ NEW

**Core Concept:**

Evaluate and mitigate cognitive biases in LLM agents making operational decisions. AIM-Bench (2025) revealed that LLM inventory managers exhibit systematic biases—the "pull-to-center" effect where agents under-order high-profit items and over-order low-profit ones, deviating from rational Newsvendor solutions.

**Why This is Novel:**

- **Literature Gap**: AIM-Bench is the first to document behavioral biases in OR agents, but only covers inventory
- **LLM-Specific**: LLMs inherit human-like cognitive biases from training data; GNNs/heuristics don't have this problem
- **Practical Impact**: Simply training to "maximize profit" is insufficient—must explicitly evaluate rationality

**MDP Formulation:**

- **State**: Market conditions, demand signals, inventory levels, historical decisions
- **Action**: Order quantities, pricing decisions, capacity allocations
- **Reward**: Profit + rationality penalty (deviation from optimal policy under known distribution)

**Proposed Benchmark: "OR-Bias-Bench"**

- **Scenarios**: Classic OR decision problems: Newsvendor, Dynamic Pricing, Revenue Management, Capacity Allocation
- **Metrics**:
  - Bias Detection Rate (identify when agent deviates from rational baseline)
  - Bias Magnitude (quantify economic loss from biased decisions)
  - Debiasing Effectiveness (improvement after intervention)

**Extension to OM:**

Extend beyond inventory to: (1) Hiring decisions with fairness constraints, (2) Resource allocation under uncertainty, (3) Multi-objective trade-offs where biases emerge in Pareto-front selection.

---

#### Direction C: Hierarchical Rule Composition for Trade Compliance ⭐ NEW

**Core Concept:**

Evaluate LLM agents' ability to apply complex, hierarchical rule systems in supply chain compliance. HSCodeComp (ICLR 2026) found a massive gap: 46.8% agent vs 95% human accuracy on HS code classification, specifically noting failures in combining multi-level rules.

**Why This is Novel:**

- **Literature Gap**: HSCodeComp is expert-level benchmark but focuses only on classification accuracy, not the reasoning process
- **LLM-Specific**: Trade compliance requires understanding legal text, applying hierarchical exceptions, and generating audit trails—fundamentally NL tasks
- **Practical Impact**: Trade compliance errors cost billions annually; automation is high-value

**MDP Formulation:**

- **State**: Product description, applicable regulations (multi-jurisdictional), precedent database, current classification attempt
- **Action**: Rule selection, exception application, classification decision, audit justification generation
- **Reward**: Classification accuracy + explanation quality + regulatory acceptance rate

**Proposed Benchmark: "OR-Compliance-Bench"**

- **Task 1**: HS Code Classification (6-digit to 10-digit refinement)
- **Task 2**: Export Control Determination (dual-use goods, sanctions screening)
- **Task 3**: Customs Valuation (transfer pricing, royalty allocation)
- **Metrics**: Rule Composition Accuracy, Exception Handling Rate, Audit Trail Quality

**Connection to PILOT-Bench:**

PILOT-Bench evaluates tool-driven workflows; OR-Compliance-Bench evaluates rule-driven workflows. Both test multi-step reasoning with verifiable outcomes (regulatory acceptance as ground truth).

---

### 5.3 TIER 2: High-Potential Directions with Partial Literature Coverage

> **TIER 2 Criteria:** Strong novelty with some existing work to build on, good MDP fit, clear differentiation from prior art.

#### Direction D: Formulation Quality & Efficiency Evaluation

**Core Concept:**

Move beyond binary correctness to evaluate the quality, efficiency, and scalability of mathematical formulations. Two correct models can differ by orders of magnitude in solve time due to formulation tightness.

**Literature Foundation:**

- **StepORLM**: Evaluates step quality but focuses on reasoning, not solver performance
- **OptiTree**: Demonstrates hierarchical decomposition yields better results—formulation structure matters
- **OptiMUS**: Notes solve time dependency on formulation but provides no systematic evaluation

**Proposed Metrics:**

- LP Relaxation Gap: |Z_IP - Z_LP| / Z_IP (tightness of continuous relaxation)
- Symmetry Score: Detection of interchangeable variables that confuse Branch & Bound
- Solve Time Ratio: Agent formulation vs. expert formulation on same problem
- Constraint Efficiency: Redundant constraint detection rate

**MDP Formulation:**

- **State**: Current formulation + solver performance metrics (node count, LP relaxation value)
- **Action**: Add symmetry-breaking constraint, tighten Big-M, reformulate objective, add cuts
- **Reward**: Solve time reduction + optimality gap improvement

---

#### Direction E: Sim-to-Real Policy Transfer for OR

**Core Concept:**

Evaluate robustness of OR policies when transferred from simulation to deployment. Sim2Act (ICLR 2026) addresses this for supply chains via adversarial calibration, but no benchmark systematically tests transfer quality.

**Why This Matters:**

- **Industrial Validation**: DeepStock deployed at Alibaba (1M+ SKUs) proves industrial scale is achievable
- **Key Barrier**: "Reality gap" between simulator and deployment remains key barrier
- **Evaluation Gap**: Need standardized evaluation of how policies degrade under distribution shift

**Proposed Benchmark: "OR-Transfer-Bench"**

- **Phase 1 (Training)**: Train policies on "clean" simulator with known dynamics
- **Phase 2 (Transfer)**: Evaluate on perturbed environments (demand noise, lead time variability, capacity shocks)
- **Metrics**: Performance degradation curve, worst-case regret, recovery speed after shock

---

#### Direction F: Causal Reasoning for Supply Chain Disruption

**Core Concept:**

Evaluate LLM agents' ability to reason about causal propagation of disruptions through supply networks. Causal-GNN SupplyNets (ICLR 2026) uses GNN-based causal world models, but LLMs offer unique advantages in interpreting unstructured disruption signals.

**LLM Advantage:**

- **Disruption Signals**: News articles, supplier communications, regulatory announcements are unstructured text
- **Causal Chains**: Understanding "Suez Canal blocked" → "semiconductor lead times increase" requires world knowledge
- **Current Practice**: Human analysts combine news + network structure + domain knowledge; LLMs can approximate this

**MDP Formulation:**

- **State**: Network topology, current disruption signals (text), inventory positions, demand forecasts
- **Action**: Expedite orders, activate backup suppliers, adjust safety stock, reroute shipments
- **Reward**: Service level maintained + cost of intervention + disruption recovery time

**Proposed Benchmark: "OR-Disruption-Bench"**

- **Scenario 1**: Semiconductor (Taiwan earthquake → fab shutdown → automotive shortage)
- **Scenario 2**: Shipping (port congestion → container shortage → retail stockouts)
- **Scenario 3**: Energy (pipeline attack → fuel shortage → logistics cost spike)
- **Metrics**: Disruption Anticipation Accuracy, Causal Chain Identification, Mitigation Effectiveness

---

### 5.4 TIER 3: Emerging Directions Requiring Further Development

> **TIER 3 Criteria:** Promising concepts with existing solutions in adjacent domains; may require more foundational work before benchmark creation.

#### Direction G: Safety-Constrained RL for Operational Decisions

**Core Concept:**

Ensure LLM agents respect hard operational constraints during exploration and deployment. Causal-GNN SupplyNets uses Lyapunov-safe RL to guarantee work-in-process limits are satisfied with high probability.

**Challenge:**

Standard RL explores freely, potentially violating safety constraints. In operations, constraint violations can be catastrophic (stockouts, safety incidents, regulatory breaches). Need RL methods that guarantee constraint satisfaction during learning.

**Current State:**

- **Methods**: Lyapunov-safe RL exists for continuous control; adaptation to LLM agents is open
- **Causal-GNN**: Focus on supply chain WIP limits; broader operational constraints unexplored
- **Gap**: Integration of constraint guarantees with LLM reasoning is novel research direction

---

#### Direction H: Multi-Agent Coordination in Service Operations

**Core Concept:**

Evaluate coordination among specialized LLM agents managing different aspects of service operations. Hospital workflow research documents multi-agent systems where bed management, staff scheduling, and patient flow agents must negotiate.

**Challenge:**

Individual agents may optimize local objectives while degrading system performance. Need evaluation of emergent coordination quality and conflict resolution.

**Proposed Evaluation Dimensions:**

- **Coordination Quality**: Nash Equilibrium Distance (how far from stable coordination)
- **Conflict Resolution Rate**: Successful negotiations / total conflicts
- **System-Level Optimality**: Overall vs. sum of individual agent objectives

---

### 5.5 Strategic Recommendation: Portfolio Approach

Rather than pursuing a single direction, we recommend a portfolio strategy that maximizes synergies:

> **Primary Submission (NeurIPS 2026):** Direction A (Solver Infeasibility Diagnosis) as the flagship contribution. This has the clearest novelty, strongest MDP formulation, and most direct PILOT-Bench continuity.

> **Secondary Submission (NeurIPS 2026 or ICML 2027):** Direction B (Behavioral Bias Detection) OR Direction C (Hierarchical Rule Composition). Both are highly novel with strong practical impact. Choose based on data availability and advisor preference.

> **Future Work (Job Talk Integration):** Position Directions D-F as the "research agenda" that extends beyond the dissertation, demonstrating breadth and long-term vision.

#### Synergy Map: How Directions Connect

| Direction Pair | Synergy | Combined Value Proposition |
|----------------|---------|----------------------------|
| A + B | Both evaluate "rationality" of LLM decisions | "Can LLMs make correct AND rational OR decisions?" |
| A + C | Both require multi-step reasoning with verification | "Sequential reasoning in constrained domains" |
| A + D | Debug = fix bad formulations; D = prevent them | "Full lifecycle of formulation quality" |
| B + F | Bias detection + causal reasoning for decisions | "Trustworthy agentic supply chain management" |
| C + F | Rule application + disruption response | "Adaptive compliance under uncertainty" |

---

## Part 6: Implementation Roadmap to NeurIPS 2026

### 6.1 Proposed Paper Structure

**Title:** "OR-Debug-Bench: A Gym Environment for Solver Infeasibility Diagnosis and Recovery"

1. **Introduction**: Motivation (solver debugging as practitioner pain); gap (no process evaluation); contribution preview
2. **Related Work**: LLM-OR benchmarks (outcome-focused), self-correction (CorrectBench gap), process rewards (StepORLM)
3. **Benchmark Design**: MDP formulation, Saboteur Agent data generation, hierarchical action space
4. **Evaluation Framework**: Recovery Rate, Diagnosis Accuracy, Optimality Preservation, Trajectory Efficiency
5. **Experiments**: Frontier LLM evaluation (GPT-4o, Claude-3.5, DeepSeek-R1); GRPO training on LLaMA-3-8B
6. **Conclusion**: Findings, implications for deployment, Supply Chain Resilience extension

### 6.2 Timeline

| Phase | Timeline | Activities | Deliverable |
|-------|----------|------------|-------------|
| 1: Data Construction | Jan-Feb 2026 | OptMATH pipeline; Saboteur Agent; 10k+ instances with ground-truth fixes | OR-Debug-Dataset-v1 |
| 2: Environment Dev | Feb-Mar 2026 | Gym interface wrapping Gurobi/Pyomo; IIS parsers; reward calculators | `pip install or-debug-bench` |
| 3: Baselines & Training | Mar-Apr 2026 | Evaluate DeepSeek-R1, GPT-4o, Claude-3.5 zero-shot; Train LLaMA-3-8B with GRPO | Results tables + learning curves |
| 4: Paper Writing | Apr-May 2026 | Draft with "Process Evaluation" narrative; position against CorrectBench, StepORLM | NeurIPS 2026 submission |

### 6.3 Resource Requirements

- **Compute**:
  - Environment: Minimal (Gurobi runs on CPU)
  - GRPO Training: ~8×A100 for full training; 2×A100 sufficient for evaluation and LoRA fine-tuning
- **Software**: Gurobi Academic License (free), Python-Gurobi, PILOT-Bench codebase as foundation
- **Data**: Public sources: MIPLIB, NL4Opt, OptMATH, IndustryOR; synthetic generation via Saboteur Agent
- **API Costs**: Frontier LLM evaluation via API (GPT-4, Claude-3, Gemini, DeepSeek-R1)

### 6.4 Alignment with NeurIPS/ICLR Trends

| Trend | Relevant Papers | Connection to OR-Debug-Bench |
|-------|-----------------|------------------------------|
| RL with Verifiable Rewards | DeepSeek-R1, OR-R1, GRPO | Solver status is ultimate verifiable reward |
| Process Reward Models | StepORLM, BiPRM, PAVs | Diagnosis Accuracy metric rewards reasoning steps |
| Agentic Supply Chain | Causal-GNN, DeepStock, C.H. Robinson | Supply Chain Resilience track extends scope to OM |
| Self-Correction Research | CorrectBench, MOID | Directly addresses 64.5% blind spot in OR context |
| Test-Time Compute | OpenAI o3, OptiTree | Metrics include Trajectory Efficiency for thinking cost |

---

## Part 7: Complete Literature Reference (70+ Papers)

### 7.1 LLM-OR Benchmarks and Datasets

| Work | Venue/Year | Scale/Focus | Key Contribution |
|------|------------|-------------|------------------|
| NL4Opt | NeurIPS 2022 | ~1,101 problems | NL to LP translation competition |
| OptiBench | arXiv 2024 | 816 problems | Formulation accuracy benchmark |
| IndustryOR | arXiv 2024 | 100 cases | Industrial complexity across 16 industries |
| OptMATH | ICML 2025 | >200k pairs | Bidirectional synthesis pipeline |
| ORQA | arXiv 2024 | 1,513 pairs | Domain knowledge QA across 20 domains |
| ComplexOR | arXiv 2024 | 37 scenarios | High ambiguity industrial cases |
| NLP4LP | arXiv 2024 | 344 problems | Verbose NL descriptions |
| PILOT-Bench | ICLR 2026 | 5,040 tasks | Probabilistic tool-driven workflows |
| MIPLIB 2017 | MPC 2021 | 1,065 instances | Standard MIP benchmark library |
| TSPLIB | ORSA 1991 | 110+ instances | Classic routing benchmark |
| HSCodeComp | ICLR 2026 | Expert-level | Hierarchical rule application (46.8% vs 95%) |
| AIM-Bench | arXiv 2025 | Inventory | Behavioral biases in LLM agents (pull-to-center) |
| CorrectBench | arXiv 2025 | Math/Code | Self-correction benchmark (64.5% blind spot) |

### 7.2 LLM-OR Systems and Frameworks

| Work | Venue/Year | Key Innovation |
|------|------------|----------------|
| OptiMUS (v0.3) | arXiv 2024 | Modular agentic OR with RAG; NLP4LP dataset |
| OptiMind | arXiv 2025 | Domain-informed error analysis; multi-turn solver feedback |
| OR-LLM-Agent | arXiv Mar 2025 | Reasoning LLMs (DeepSeek-R1) for OR |
| OR-R1 | arXiv 2025 | Task-specific GRPO; 6%+ improvement on OR tasks |
| Chain-of-Experts | ICLR 2024 | Multi-agent collaboration for OR subtasks |
| LinearizeLLM | arXiv 2025 | Agent-based linearization of nonlinear constraints |
| HeuriGym | OpenReview 2025 | Agentic benchmark for LLM-crafted heuristics |
| OptiTree | NeurIPS 2025 | Hierarchical tree search for optimization modeling |
| LEAN-LLM-OPT | arXiv 2024 | Few-shot adaptation to large-scale optimization |
| MOID | arXiv 2025 | Multi-objective infeasibility diagnosis for routing |

### 7.3 Reinforcement Learning and Process Supervision

| Work | Venue/Year | Key Contribution |
|------|------------|------------------|
| DeepSeek-R1 | Nature Jan 2025 | Pure RL induces reasoning; GRPO algorithm |
| GRPO Analysis | arXiv 2025 | Effective loss dynamics and success amplification |
| Tülu 3 (RLVR) | Allen AI 2024 | Reinforcement learning with verifiable rewards |
| DeepSeekMath | arXiv 2024 | GRPO for math reasoning; 51.7% on MATH |
| StepORLM | OpenReview 2025 | GenPRM for OR; dual-feedback mechanism |
| BiPRM | arXiv 2025 | Bidirectional process reward model |
| Math-Shepherd | ACL 2024 | Automatic process supervision via MC estimation |
| PAVs | ICLR 2025 | Process Advantage Verifiers; 6× efficiency |
| ThinkPRM | OpenReview 2025 | Generative verification with 1% training data |
| TaTToo | arXiv 2025 | Tool-grounded thinking PRM |

### 7.4 Operations Management and Supply Chain

| Work | Venue/Year | Key Contribution |
|------|------------|------------------|
| C.H. Robinson Agentic | Industry 2025 | Industrial deployment of autonomous supply chain agents |
| Causal-GNN SupplyNets | ICLR 2026 | Lyapunov-safe RL for semiconductor supply chains |
| Sim2Act | ICLR 2026 | Adversarial calibration for sim-to-real transfer |
| DeepStock | NeurIPS MLxOR 2025 | Policy regularization; deployed at Alibaba (1M+ SKUs) |
| Create-to-Reuse | ICLR 2026 | Dynamic action space expansion for service operations |
| Multi-Agent Hospital | Medium 2025 | Agentic AI workflows for patient flow |
| Online Decision w/ Generative Actions | ICLR 2026 | Generative action sets for online learning |

### 7.5 Self-Correction, Reasoning, and Behavioral AI

| Work | Venue/Year | Key Finding |
|------|------------|-------------|
| CorrectBench | arXiv 2025 | Self-correction benchmark; 64.5% blind spot rate |
| Self-Correction Blind Spot | OpenReview 2025 | Models fix external but not internal errors |
| Self-Refine | NeurIPS 2023 | 20% improvement through iterative refinement |
| Anthropic Faithfulness | Anthropic 2024 | Larger models produce less faithful reasoning |
| FaithCoT-Bench | arXiv 2024 | Unfaithfulness detection benchmark |
| AIM-Bench Biases | arXiv 2025 | Pull-to-center effect in inventory decisions |

### 7.6 Agent Evaluation and Test-Time Compute

| Work | Venue/Year | Key Contribution |
|------|------------|------------------|
| AgentBench | ICLR 2024 | Comprehensive agent evaluation; GPT-4 at 52.47% |
| WorfBench | arXiv 2024 | Workflow robustness evaluation |
| SWE-bench | ICLR 2024 | Software engineering agent evaluation |
| OpenAI o3 | OpenAI Jan 2026 | Test-time compute scaling; 71.7% on SWE-bench Verified |
| Scaling Test-Time Compute | ICLR 2025 | Smaller models outperform 14× larger with optimal inference |
| AlphaProof | DeepMind 2024 | MCTS + formal verification for mathematics |

### 7.7 Trade Compliance and Hierarchical Rules

| Work | Venue/Year | Key Contribution |
|------|------------|------------------|
| HSCodeComp | ICLR 2026 | Expert-level HS code classification; 46.8% agent accuracy |
| Trade Compliance AI | Industry Reports | Growing market for automated classification |
| Hierarchical Rule Systems | Legal AI Research | Multi-level exception handling in regulations |

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

### Job Talk Narrative Arc (Enhanced)

1. **Chapter 1 (Foundation):** PILOT-Bench established the foundation for evaluating LLM agents under uncertainty, introducing MDP perspective to workflow execution.
2. **Chapter 2 (Verification):** OR-Debug-Bench extends to mathematical optimization debugging, leveraging solver-backed verification to overcome the Self-Correction Blind Spot.
3. **Chapter 3 (Rationality & Compliance):** OR-Bias-Bench / OR-Compliance-Bench expand to behavioral rationality and regulatory compliance, demonstrating breadth across OR/OM.
4. **Chapter 4 (Future Vision):** Future directions in disruption reasoning, sim-to-real transfer, and safety-constrained RL complete the vision of trustworthy agentic operations.

### Immediate Next Steps

1. **Direction A:** Finalize OR-Debug-Bench MDP specification with Saboteur Agent error taxonomy
2. **Direction B:** Scope OR-Bias-Bench: Identify 3-5 classic OM decision problems with known optimal policies for bias evaluation
3. **Direction C:** Evaluate HSCodeComp feasibility: Assess data availability and regulatory expertise needed for rule composition benchmark
4. **Infrastructure:** Prototype data generation infrastructure using OptMATH + MIPLIB with controlled error injection
5. **Validation:** Baseline evaluation on DeepSeek-R1, GPT-4o, Claude-3.5 to validate benchmark discriminability across directions

---

*End of Report*
