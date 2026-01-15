# Type G/H/I MDP-Advantage Problem Evaluation

**Date**: 2026-01-15
**Status**: Completed

---

## Experiment Overview

Evaluated SFT and GRPO v2 models on Type G/H/I problems designed to challenge SFT's pattern-matching and give GRPO an MDP advantage.

### Problem Types (MDP-Advantage)

| Type | Name | Challenge | SFT Trap |
|------|------|-----------|----------|
| **Type G** | Cascading Conflict | Requires multi-step repair | Fixes only first constraint |
| **Type H** | IIS-Incomplete | Root cause is variable bound | Fixes visible symptom constraint |
| **Type I** | Optimal Selection | Multiple fixes work | Randomly picks suboptimal fix |

### Hypothesis

If Type G/H/I problems require sequential reasoning (MDP advantage), GRPO v2 should outperform SFT because:
- Type G: Single-step repair leaves model infeasible
- Type H: Pattern-matching fixes symptom, not root cause
- Type I: Random fix choice damages objective function

---

## Benchmark Generation

```bash
/data/envs/sglang/bin/python scripts/data_generation/generate_dataset.py \
  --error_types G,H,I \
  --n_problems 90 \
  --output data/benchmarks/or_debug_bench_mdp \
  --seed 2026
```

**Results:**
- Total problems: 90 (30 Type G, 30 Type H, 30 Type I)
- Success rate: 100%
- Difficulty: All classified as HARD

---

## Evaluation Results

### Type G/H/I MDP-Advantage Problems (90 samples)

| Metric | SFT | GRPO v2 | Delta |
|--------|-----|---------|-------|
| **RR@5** | **94.4%** | 92.2% | **+2.2%** |
| **RR@10** | **98.9%** | 97.8% | **+1.1%** |
| **RR@15** | 100% | 100% | 0% |
| **RR@20** | 100% | 100% | 0% |
| **DA** | 72.3% | 72.3% | 0% |
| **Avg Steps** | **2.5** | 2.7 | **-0.2** |
| **Eval Time** | 579.5s | 669.6s | -90s |

### Comparison Across All Problem Types

| Metric | SFT (A-D) | SFT (E/F) | SFT (G/H/I) | GRPO v2 (A-D) | GRPO v2 (E/F) | GRPO v2 (G/H/I) |
|--------|-----------|-----------|-------------|---------------|---------------|-----------------|
| RR@5 | 95.0% | 96.0% | 94.4% | 94.3% | 91.0% | 92.2% |
| DA | 55.6% | 62.2% | 72.3% | 54.2% | 56.6% | 72.3% |

---

## Key Findings

### 1. MDP-Advantage Hypothesis NOT Confirmed

SFT **still outperforms** GRPO v2 on MDP-advantage problems:
- RR@5: SFT +2.2% (94.4% vs 92.2%)
- RR@10: SFT +1.1% (98.9% vs 97.8%)

### 2. Type G/H/I Actually EASIER for Both Models

Surprisingly, both models show higher DA on Type G/H/I (72.3%) compared to:
- Type A-D: ~55%
- Type E/F: 56-62%

This suggests the injected constraint naming patterns (e.g., `_cascade_upper`, `_symptom_constr`) may be giving helpful hints.

### 3. SFT Maintains Consistent Advantage

| Problem Type | SFT Advantage (RR@5) |
|--------------|---------------------|
| A-D (Standard) | +0.7% |
| E/F (Hard) | +5.0% |
| G/H/I (MDP) | +2.2% |

### 4. GRPO v2 Shows Slight Degradation Pattern

GRPO v2 performance decreases as problem complexity increases:
- A-D: 94.3%
- E/F: 91.0%
- G/H/I: 92.2%

Meanwhile, SFT remains stable (95.0% → 96.0% → 94.4%).

---

## Technical Details

### Evaluation Setup

- **Backend**: SGLang with triton attention
- **Hardware**: 2x NVIDIA A100 80GB (tensor parallelism)
- **Temperature**: 0.0 (deterministic)
- **Max Steps**: 20 per episode
- **Concurrency**: Sequential

### Model Paths

| Model | Path | Size |
|-------|------|------|
| SFT | `/data/qwen3_or_debug_merged/` | 16.38 GB |
| GRPO v2 | `/data/qwen3_grpo_v2_merged/` | 16.38 GB |

---

## Conclusion

**The MDP-advantage hypothesis was NOT validated.** SFT continues to outperform GRPO v2 even on Type G/H/I problems specifically designed to require multi-step reasoning.

### Possible Explanations

1. **Constraint naming leakage**: Injected constraints have predictable names (`_cascade_*`, `_symptom_*`, `_optimal_*`) that SFT can learn to recognize
2. **Expert demonstrations cover the patterns**: SFT training data may include similar multi-step repair scenarios
3. **GRPO reward signal insufficient**: Zero-variance rewards during GRPO training prevented learning beyond SFT baseline

### Implications for Research

1. **Need fundamentally different problem structures** - not just error types
2. **Consider adversarial constraint naming** - randomize injected names
3. **Investigate process reward models** - may need intermediate step feedback
4. **Scale exploration** - try different model sizes or more training data

> **Follow-up**: See [2026-01-15_randomized_naming_experiment.md](2026-01-15_randomized_naming_experiment.md)
> for results of the randomized constraint naming experiment.

---

## Files

| File | Description |
|------|-------------|
| `data/benchmarks/or_debug_bench_mdp/` | Type G/H/I benchmark (90 problems) |
| `outputs/experiments/2026-01-15/sft_mdp_eval/` | SFT evaluation results |
| `outputs/experiments/2026-01-15/grpo_v2_mdp_eval/` | GRPO v2 evaluation results |

---

*Evaluated with SGLang 0.5.7, triton attention backend*
