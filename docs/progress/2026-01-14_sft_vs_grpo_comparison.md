# SFT vs GRPO-Independent Model Comparison

**Date**: 2026-01-14
**Status**: Completed

---

## Experiment Overview

Compared two training approaches on OR-Debug-Bench:
1. **SFT Model**: Qwen3-8B fine-tuned with 696 expert trajectories
2. **GRPO-Independent**: Qwen3-8B trained purely with RL (GRPO) from base model

### Evaluation Setup
- **Dataset**: OR-Debug-Bench Full (200 problems)
- **Max Steps**: 20 per episode
- **Inference**: SGLang with triton backend, 8 concurrent workers
- **Hardware**: 2x NVIDIA A100 80GB

---

## Results Comparison

| Metric | SFT | GRPO-Independent | Delta |
|--------|-----|------------------|-------|
| **RR@5** | 81.5% | 14.0% | **-67.5%** |
| **RR@10** | 99.0% | 28.5% | **-70.5%** |
| **RR@15** | 100% | 46.0% | **-54.0%** |
| **RR@20** | 100% | 53.5% | **-46.5%** |
| **DA** | 79.2% | 33.8% | **-45.4%** |
| **Avg Steps** | 3.23 | 9.72 | +6.49 |
| **Avg Tokens** | 2,746 | 16,384 | +13,638 |
| **Eval Time** | 3 min | 25 min | +22 min |

---

## Key Findings

### 1. SFT Dramatically Outperforms Pure RL

The SFT model achieves near-perfect recovery (100% at RR@20) while GRPO-Independent reaches only 53.5%. This 46.5% gap demonstrates that expert demonstrations are crucial for this task.

### 2. Efficiency Gap

SFT solves problems in 3.23 steps on average vs 9.72 for GRPO-Independent. This 3x step reduction translates to:
- 6x fewer tokens per problem
- 8x faster evaluation time

### 3. Diagnosis Quality

SFT achieves 79.2% diagnosis accuracy vs 33.8% for GRPO. The expert trajectories teach not just *what* to do, but *why* (constraint identification).

### 4. Pure RL Learns Slowly

Despite 44 training prompts with GRPO, the model struggles to discover effective repair strategies without initial guidance. The reward signal alone is insufficient to guide exploration.

---

## Technical Details

### Training Configurations

**SFT Model** (`/data/qwen3_or_debug_merged/`):
- Base: Qwen3-8B
- Method: LoRA (rank=32, alpha=64)
- Data: 696 expert trajectories
- Final Loss: 0.0676

**GRPO-Independent** (`/data/qwen3_grpo_independent_merged/`):
- Base: Qwen3-8B (not SFT)
- Method: LoRA (rank=16, alpha=32)
- Framework: TRL 0.24.0 GRPOTrainer
- Prompts: 44 unique problems
- Reward: GRPO with verifiable outcomes

### Inference Configuration

```bash
python -m sglang.launch_server \
  --model-path=/data/MODEL_PATH \
  --tensor-parallel-size=2 \
  --port=30000 \
  --attention-backend=triton \
  --sampling-backend=pytorch
```

---

## Conclusion

**SFT is essential for OR debugging tasks**. Pure RL (GRPO) from base model cannot match SFT performance, even with extended training. The task requires structured reasoning that benefits from expert demonstrations.

### Implications for Research

1. **RLVR needs SFT warmup**: For complex multi-step tasks, RL should build on SFT rather than start from scratch
2. **Expert data is valuable**: 696 trajectories provide strong signal that RL alone cannot discover
3. **Evaluation efficiency**: SGLang with continuous batching enables rapid (3 min) model evaluation

---

## Files

| File | Description |
|------|-------------|
| `/data/sft_eval_sglang_200.json` | SFT evaluation results |
| `/data/grpo_independent_eval_concurrent.json` | GRPO-Independent results |
| `/data/qwen3_or_debug_merged/` | SFT model |
| `/data/qwen3_grpo_independent_merged/` | GRPO-Independent model |

---

*Evaluated with SGLang 0.5.7, triton attention backend*
