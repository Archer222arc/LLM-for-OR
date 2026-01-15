# GRPO v2 Full-Scale Evaluation

**Date**: 2026-01-15
**Status**: Completed

---

## Experiment Overview

Full-scale evaluation of GRPO v2 model trained with hybrid rewards on the complete held-out test set, compared against the SFT baseline.

### Key Improvements in GRPO v2

GRPO v2 addresses the zero-variance reward problem from v1 by implementing a **hybrid reward strategy**:

```
R = R_outcome + R_process + R_heuristic
```

- **R_outcome**: Binary success/failure from Gurobi verification
- **R_process**: Step-wise progress signals (IIS reduction, constraint identification)
- **R_heuristic**: Small bonus for structured reasoning patterns

This achieved non-zero reward variance (reward_std ~1.0) enabling gradient updates.

### Evaluation Setup

- **Dataset**: OR-Debug-Bench Held-out (4,162 problems, excluding 300 training samples)
- **Max Steps**: 20 per episode
- **Inference**: SGLang with triton backend, 8 concurrent workers
- **Hardware**: 2x NVIDIA A100 80GB (tensor parallelism)
- **Evaluation Mode**: Sequential (memory constraints)

---

## Results Comparison

| Metric | SFT | GRPO v2 | Delta |
|--------|-----|---------|-------|
| **RR@5** | 95.0% | 94.3% | -0.7% |
| **RR@10** | 99.4% | 99.2% | -0.2% |
| **RR@20** | 100% | 100% | 0% |
| **DA** | 55.6% | 54.2% | -1.4% |
| **Avg Steps** | 1.9 | 2.0 | +0.1 |

---

## Key Findings

### 1. SFT Slightly Outperforms GRPO v2

Despite achieving non-zero reward variance, GRPO v2 does not improve upon SFT. The SFT model shows marginally better performance across all metrics.

### 2. Both Models Achieve Perfect RR@20

Both models eventually solve 100% of problems within 20 steps, demonstrating robust repair capabilities.

### 3. High Efficiency Maintained

Average steps remain low (1.9-2.0) for both models, indicating efficient repair strategies learned from SFT training.

### 4. Generalization Confirmed

Testing on held-out data (excluding 300 training samples) confirms no overfitting. Results on held-out set match full benchmark performance.

---

## Technical Details

### Model Configurations

**SFT Model** (`/data/qwen3_or_debug_merged/`):
- Base: Qwen3-8B
- Method: LoRA (rank=32, alpha=64)
- Data: 696 expert trajectories
- Final Loss: 0.0676
- Size: 16.38 GB

**GRPO v2 Model** (`/data/qwen3_grpo_v2_merged/`):
- Base: SFT model (continued training)
- Method: LoRA (rank=16, alpha=32)
- Framework: TRL 0.24.0 GRPOTrainer
- Prompts: 300 problems
- Reward: Hybrid (outcome + process + heuristic)
- Size: 16.38 GB

### Held-out Test Set Creation

Created held-out benchmark by excluding training samples:

```bash
python scripts/evaluation/create_holdout_test.py \
  --benchmark data/benchmarks/or_debug_bench_full \
  --exclude data/training/grpo_training_ids.txt \
  --output data/benchmarks/or_debug_bench_holdout
```

Result: 4,162 held-out problems (excluded 300 training samples)

### Inference Configuration

```bash
python -m sglang.launch_server \
  --model-path /data/MODEL_PATH \
  --tensor-parallel-size 2 \
  --port 30000 \
  --attention-backend triton \
  --disable-cuda-graph
```

---

## Conclusion

**GRPO v2 does not improve over SFT despite non-zero reward variance**. The SFT model already achieves near-optimal performance on OR debugging tasks, leaving little room for RL improvement.

### Implications for Research

1. **SFT ceiling effect**: When SFT achieves >95% RR@5, additional RL provides diminishing returns
2. **Reward design complexity**: Even with hybrid rewards, RL struggles to find better policies than expert demonstrations
3. **Held-out validation**: Confirmed no overfitting to training problems

### Future Directions

1. **Harder benchmarks**: Test on problems where SFT struggles (DA < 80%)
2. **Process supervision**: Use step-wise rewards more aggressively
3. **Different RL algorithms**: Try PPO, DPO, or other methods

---

## Files

| File | Description |
|------|-------------|
| `/data/benchmarks/or_debug_bench_holdout/` | Held-out test benchmark |
| `/data/training/grpo_training_ids.txt` | Training problem IDs (300) |
| `/data/qwen3_or_debug_merged/` | SFT model |
| `/data/qwen3_grpo_v2_merged/` | GRPO v2 model |
| `scripts/evaluation/create_holdout_test.py` | Held-out set creation script |

---

*Evaluated with SGLang 0.5.7, triton attention backend*
