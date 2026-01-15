# Type E/F Hard Problem Evaluation

**Date**: 2026-01-15
**Status**: Completed

---

## Experiment Overview

Evaluated SFT and GRPO v2 models on harder Type E/F problems to test the "SFT ceiling effect" hypothesis.

### Problem Types

| Type | Name | Challenge |
|------|------|-----------|
| **Type E** | Multi-Constraint Conflict | Requires fixing 2+ constraints simultaneously |
| **Type F** | Hidden Dependency | Root cause not directly visible in IIS |

### Hypothesis

If Type E/F problems are sufficiently difficult, SFT's RR@5 may drop to 60-80%, giving GRPO v2 an opportunity to demonstrate advantage.

---

## Benchmark Generation

```bash
python scripts/data_generation/generate_dataset.py \
  --error_types E,F \
  --n_problems 100 \
  --output data/benchmarks/or_debug_bench_hard \
  --seed 2026
```

**Results:**
- Total problems: 100 (50 Type E, 50 Type F)
- Success rate: 100%
- Difficulty: All classified as HARD

---

## Evaluation Results

### Type E/F Hard Problems (100 samples)

| Metric | SFT | GRPO v2 | Delta |
|--------|-----|---------|-------|
| **RR@5** | **96.0%** | 91.0% | **+5.0%** |
| **RR@10** | **100%** | 99.0% | **+1.0%** |
| **RR@15** | 100% | 100% | 0% |
| **RR@20** | 100% | 100% | 0% |
| **DA** | **62.2%** | 56.6% | **+5.6%** |
| **Avg Steps** | 2.2 | 2.2 | 0 |
| **Eval Time** | 26.6 min | 30.2 min | +3.6 min |

### Comparison: Easy (A-D) vs Hard (E/F)

| Metric | SFT Easy | SFT Hard | GRPO v2 Easy | GRPO v2 Hard |
|--------|----------|----------|--------------|--------------|
| RR@5 | 95.0% | 96.0% | 94.3% | 91.0% |
| DA | 55.6% | 62.2% | 54.2% | 56.6% |

---

## Key Findings

### 1. Hypothesis NOT Confirmed

SFT did **NOT** hit a performance ceiling on Type E/F problems. Performance actually improved slightly:
- RR@5: 96.0% (hard) vs 95.0% (easy)
- DA: 62.2% (hard) vs 55.6% (easy)

### 2. SFT Maintains Advantage

SFT outperforms GRPO v2 on hard problems with an even **wider gap**:
- Easy problems: SFT +0.7% RR@5
- Hard problems: SFT +5.0% RR@5

### 3. GRPO v2 Degraded on Hard Problems

GRPO v2 showed slight performance degradation:
- RR@5: 91.0% (hard) vs 94.3% (easy) = -3.3%

### 4. Type E/F May Not Be "Harder" for SFT

Expert demonstrations from SFT training generalize well to:
- Multi-constraint conflicts (Type E)
- Hidden dependencies (Type F)

---

## Technical Details

### Evaluation Setup

- **Backend**: SGLang with triton attention
- **Hardware**: 2x NVIDIA A100 80GB (tensor parallelism)
- **Temperature**: 0.0 (deterministic)
- **Max Steps**: 20 per episode
- **Concurrency**: Sequential (memory constraints)

### Model Paths

| Model | Path | Size |
|-------|------|------|
| SFT | `/data/qwen3_or_debug_merged/` | 16.38 GB |
| GRPO v2 | `/data/qwen3_grpo_v2_merged/` | 16.38 GB |

---

## Conclusion

**The "SFT ceiling effect" hypothesis was not validated.** SFT maintains robust performance across all error types, including the harder Type E/F problems.

### Implications

1. **Expert demonstrations generalize well**: SFT training on simpler problems transfers to harder cases
2. **GRPO v2 hybrid rewards insufficient**: Even with non-zero variance, RL doesn't improve upon SFT
3. **Need different problem structures**: May require fundamentally different challenges (not just error types) to expose SFT limitations

### Future Directions

1. Test on problems where SFT struggles (DA < 50%)
2. Try larger model scale differences
3. Investigate multi-turn reasoning requirements

---

## Files

| File | Description |
|------|-------------|
| `data/benchmarks/or_debug_bench_hard/` | Type E/F hard benchmark (100 problems) |
| `outputs/experiments/2026-01-15/sft_hard_eval/` | SFT evaluation results |
| `outputs/experiments/2026-01-15/grpo_v2_hard_eval/` | GRPO v2 evaluation results |

---

*Evaluated with SGLang 0.5.7, triton attention backend*
