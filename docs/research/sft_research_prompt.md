# SFT Research Request: Fine-tuning Qwen3-8B for OR Debugging

## Project Context

I'm building **OR-Debug-Bench**, a benchmark for evaluating LLM agents on Operations Research model debugging. The agent receives an infeasible optimization model and must diagnose/repair it through multi-step interactions with a Gurobi solver.

### Task Format (MDP)
- **State**: Problem description + solver status + IIS (Irreducible Infeasible Subsystem)
- **Action**: GET_IIS, DROP_CONSTRAINT, RELAX_CONSTRAINT, SUBMIT
- **Goal**: Restore model feasibility

---

## Current SFT Data

### Dataset Statistics
| Source | Samples | Success Rate |
|--------|---------|--------------|
| gpt-5.2-chat | 774 | 77.4% |
| HeuristicAgent | 4,442 | 100% |
| **Total** | **5,216** | - |

### Data Format (Alpaca-style)
```json
{
  "instruction": "Debug the infeasible optimization model and provide the action to fix it.",
  "input": "## Problem\nID: mip_typeA_000\nType: A\n\n## Current State\nStatus: INFEASIBLE\n\n## IIS (Irreducible Infeasible Subsystem)\nConflicting Constraints: ['c1', 'c_key_upper']\nConflicting Bounds: ['x1', 'x2', 'x3']\n\n## Model Structure\nConstraints: 11\nVariables: 9",
  "output": "<think>\nStep 4: Constraint 'c_key_upper' is too tight.\nRelaxing by 1.0 to allow feasibility.\n</think>\n\nAction: RELAX_CONSTRAINT(c_key_upper, 1.0)",
  "metadata": {
    "problem_id": "mip_typeA_000",
    "error_type": "A",
    "steps": 4,
    "agent": "SFT-Teacher-gpt-5.2-chat"
  }
}
```

### Key Characteristics
1. **Reasoning format**: Uses `<think>...</think>` tags (similar to DeepSeek-R1)
2. **Structured output**: Action must follow specific format
3. **Short sequences**: Input ~200-500 tokens, Output ~50-150 tokens
4. **Domain-specific**: OR/optimization terminology

---

## Target Model

**Qwen/Qwen3-8B-Instruct** (or Qwen2.5-7B-Instruct as fallback)

### Compute Resources
- **GPU**: 2× NVIDIA A800 (80GB each)
- **Framework preference**: PyTorch + HuggingFace ecosystem
- **Training time budget**: ~24 hours

---

## Research Questions

### 1. Framework Selection
Compare these SFT frameworks for my use case:
- **LLaMA-Factory**: ease of use, Qwen support?
- **Axolotl**: flexibility, multi-GPU?
- **TRL (SFTTrainer)**: official HF, simplicity?
- **ms-swift**: Alibaba's tool for Qwen?
- **Unsloth**: speed optimization?

Criteria:
- Native Qwen3 support
- Multi-GPU (2×A800) support
- LoRA/QLoRA support
- Custom chat template handling

### 2. Training Configuration
Recommend optimal hyperparameters:
- LoRA rank/alpha for 8B model
- Batch size for 2×A800 (160GB total)
- Learning rate schedule
- Number of epochs for ~5K samples
- DeepSpeed ZeRO stage?

### 3. Data Format Conversion
How to convert my Alpaca-style data to:
- Qwen chat format (with `<think>` preservation)
- ShareGPT format (if needed)
- Handle special tokens properly

### 4. Evaluation Strategy
- How to evaluate SFT model before RLVR stage?
- Metrics beyond loss: generation quality, format compliance?
- Validation split strategy for 5K samples?

### 5. Common Pitfalls
What are common failure modes when:
- Fine-tuning reasoning models (with `<think>` tags)?
- Using LoRA on instruction-tuned base?
- Training on domain-specific (OR) data?

---

## Specific Constraints

1. **Must preserve `<think>` reasoning** - This is crucial for later GRPO/RLVR training
2. **Output format compliance** - Action must be parseable: `ACTION_TYPE(args)`
3. **No catastrophic forgetting** - Model should retain general capabilities
4. **Reproducibility** - Need to document exact config for paper

---

## Desired Output

Please provide:
1. **Recommended framework** with justification
2. **Complete training config** (YAML or Python)
3. **Data conversion script** (if format change needed)
4. **Training command** for 2×A800 setup
5. **Evaluation checklist** before deployment

---

## References

Related work I'm aware of:
- DeepSeek-R1: GRPO for reasoning
- Qwen technical reports
- LoRA/QLoRA papers
- LLaMA-Factory documentation

Please search for the latest (2024-2025) best practices for SFT on reasoning tasks.
