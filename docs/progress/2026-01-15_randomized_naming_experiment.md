# Randomized Constraint Naming Experiment

**Date**: 2026-01-15
**Status**: Completed
**Prior**: [2026-01-15_type_ghi_mdp_evaluation.md](2026-01-15_type_ghi_mdp_evaluation.md)

---

## Objective

Test whether SFT's strong performance on Type G/H/I (MDP-advantage) problems was due to pattern-matching on predictable constraint names.

## Hypothesis

SFT might be "cheating" by recognizing injected constraint patterns like `_cascade_upper_x0`, `_symptom_constr_x2`, `_optimal_lower_x0` rather than learning genuine debugging strategies. Randomizing these names should expose this limitation.

## Methodology

### Old Benchmark (Predictable Names)
- Type G: `_cascade_upper_x{n}`, `_cascade_lower_x{n}`
- Type H: `_symptom_constr_x{n}`
- Type I: `_optimal_upper_x{n}`, `_optimal_lower_x{n}`

### New Benchmark (Randomized Names)
- Type G: `c_{6hex}_ub`, `c_{6hex}_lb` (e.g., `c_99d99b_ub`)
- Type H: `constr_{6hex}` (e.g., `constr_113e75`)
- Type I: `bound_{6hex}_l`, `bound_{6hex}_u` (e.g., `bound_4f7895_l`)

## Results

### Old Benchmark (`or_debug_bench_mdp`)

| Model | RR@5 | RR@10 | DA | Avg Steps |
|-------|------|-------|-----|-----------|
| SFT | 94.4% | 98.9% | 72.3% | 2.5 |
| GRPO v2 | 92.2% | 97.8% | 72.3% | 2.7 |
| **Diff** | **+2.2%** | **+1.1%** | **0.0%** | **-0.2** |

### New Benchmark (`or_debug_bench_mdp_v2`)

| Model | RR@5 | RR@10 | DA | Avg Steps |
|-------|------|-------|-----|-----------|
| SFT | 95.6% | 98.9% | 73.3% | 2.5 |
| GRPO v2 | 88.9% | 100.0% | 72.4% | 2.7 |
| **Diff** | **+6.7%** | **-1.1%** | **+0.9%** | **-0.2** |

### Model Performance Change

| Model | RR@5 Change | DA Change |
|-------|-------------|-----------|
| SFT | +1.2% (94.4% → 95.6%) | +1.0% |
| GRPO v2 | -3.3% (92.2% → 88.9%) | +0.1% |

## Key Findings

### 1. SFT is NOT Pattern-Matching
- **Contrary to hypothesis**, SFT performance **improved** with randomized names (+1.2%)
- This suggests SFT learned genuine debugging strategies, not constraint name patterns
- The slight improvement may be due to benchmark variance or more diverse training

### 2. GRPO v2 Performance Dropped
- GRPO v2 showed a **3.3% decrease** in RR@5 with randomized names
- Surprisingly, GRPO may have been the one benefiting from predictable patterns
- This could indicate GRPO's RL signal was partially capturing naming patterns

### 3. SFT Advantage Widened
- SFT's advantage over GRPO increased from **+2.2%** to **+6.7%** in RR@5
- The MDP-advantage hypothesis (that GRPO should excel on multi-step problems) is **NOT supported**

## Conclusions

1. **SFT's strong performance is genuine** - it learned to debug OR problems effectively, not just memorize patterns

2. **MDP-advantage for GRPO not observed** - On problems requiring multi-step reasoning (Type G/H/I), SFT actually outperforms GRPO

3. **Possible explanations**:
   - SFT training data quality was high (expert demonstrations)
   - GRPO's reward signal may have been too sparse or noisy
   - The problem complexity may not be sufficient to demonstrate GRPO's advantage

## Files

| File | Description |
|------|-------------|
| `data/benchmarks/or_debug_bench_mdp_v2/` | New benchmark with randomized names |
| `outputs/experiments/2026-01-15/2026-01-15/sft_mdp_v2_eval/` | SFT results on new benchmark |
| `outputs/experiments/2026-01-15/2026-01-15/grpo_v2_mdp_v2_eval/` | GRPO v2 results on new benchmark |

---

## Technical Details

### Benchmark Generation

```bash
/data/envs/sglang/bin/python scripts/data_generation/generate_dataset.py \
  --error_types G,H,I \
  --n_problems 90 \
  --output data/benchmarks/or_debug_bench_mdp_v2 \
  --seed 20260115
```

### Evaluation Setup

- **Backend**: SGLang with triton attention (flashinfer disabled)
- **Hardware**: 2x NVIDIA A100 80GB (tensor parallelism)
- **Temperature**: 0.0 (deterministic)
- **Max Steps**: 20 per episode

### Model Paths

| Model | Path |
|-------|------|
| SFT | `/data/qwen3_or_debug_merged/` |
| GRPO v2 | `/data/qwen3_grpo_v2_merged/` |

---

## Next Steps

1. Investigate why GRPO v2 performance dropped with randomization
2. Consider increasing problem complexity to test MDP-advantage hypothesis
3. Analyze per-problem-type performance breakdown

---

*Evaluated with SGLang 0.5.7, triton attention backend*
