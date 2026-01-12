# Complete Literature Reference (70+ Papers)

## 7.1 LLM-OR Benchmarks and Datasets

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

---

## 7.2 LLM-OR Systems and Frameworks

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

---

## 7.3 Reinforcement Learning and Process Supervision

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

---

## 7.4 Operations Management and Supply Chain

| Work | Venue/Year | Key Contribution |
|------|------------|------------------|
| C.H. Robinson Agentic | Industry 2025 | Industrial deployment of autonomous supply chain agents |
| Causal-GNN SupplyNets | ICLR 2026 | Lyapunov-safe RL for semiconductor supply chains |
| Sim2Act | ICLR 2026 | Adversarial calibration for sim-to-real transfer |
| DeepStock | NeurIPS MLxOR 2025 | Policy regularization; deployed at Alibaba (1M+ SKUs) |
| Create-to-Reuse | ICLR 2026 | Dynamic action space expansion for service operations |
| Multi-Agent Hospital | Medium 2025 | Agentic AI workflows for patient flow |
| Online Decision w/ Generative Actions | ICLR 2026 | Generative action sets for online learning |

---

## 7.5 Self-Correction, Reasoning, and Behavioral AI

| Work | Venue/Year | Key Finding |
|------|------------|-------------|
| CorrectBench | arXiv 2025 | Self-correction benchmark; 64.5% blind spot rate |
| Self-Correction Blind Spot | OpenReview 2025 | Models fix external but not internal errors |
| Self-Refine | NeurIPS 2023 | 20% improvement through iterative refinement |
| Anthropic Faithfulness | Anthropic 2024 | Larger models produce less faithful reasoning |
| FaithCoT-Bench | arXiv 2024 | Unfaithfulness detection benchmark |
| AIM-Bench Biases | arXiv 2025 | Pull-to-center effect in inventory decisions |

---

## 7.6 Agent Evaluation and Test-Time Compute

| Work | Venue/Year | Key Contribution |
|------|------------|------------------|
| AgentBench | ICLR 2024 | Comprehensive agent evaluation; GPT-4 at 52.47% |
| WorfBench | arXiv 2024 | Workflow robustness evaluation |
| SWE-bench | ICLR 2024 | Software engineering agent evaluation |
| OpenAI o3 | OpenAI Jan 2026 | Test-time compute scaling; 71.7% on SWE-bench Verified |
| Scaling Test-Time Compute | ICLR 2025 | Smaller models outperform 14× larger with optimal inference |
| AlphaProof | DeepMind 2024 | MCTS + formal verification for mathematics |

---

## 7.7 Trade Compliance and Hierarchical Rules

| Work | Venue/Year | Key Contribution |
|------|------------|------------------|
| HSCodeComp | ICLR 2026 | Expert-level HS code classification; 46.8% agent accuracy |
| Trade Compliance AI | Industry Reports | Growing market for automated classification |
| Hierarchical Rule Systems | Legal AI Research | Multi-level exception handling in regulations |

---

*Total Papers Reviewed: 70+*

*Next: [04_IMPLEMENTATION_ROADMAP.md](04_IMPLEMENTATION_ROADMAP.md) - Timeline and Resources*
