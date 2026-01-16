# DGRO Training Results

**Date**: 2026-01-16
**Status**: Completed
**Prior**: [2026-01-15_phase2_grpo_improvements.md](2026-01-15_phase2_grpo_improvements.md)

---

## Objective

Train DGRO (Decoupled Group Reward Optimization) to address the reward variance collapse problem observed in GRPO training.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `/data/qwen3_or_debug_merged` (SFT) |
| β_kl | 0.001 |
| β_reward | 2.0 |
| Min Variance | 0.01 |
| Temperature | 1.0 |
| Num Generations | 4 |
| Num Epochs | 3 |
| Training Samples | 300 |
| Total Steps | 450 |

## Training Statistics

| Metric | Value |
|--------|-------|
| Total Reward Computations | 450 |
| Mean Reward | -49.99 |
| Std Reward | 1.1516 |
| **Mean Variance** | **1.344** |
| Low Variance Rate | 0.0% |
| Training Time | 5.3 hours |

**Key Finding**: DGRO successfully maintained variance > 0, unlike standard GRPO which collapsed to zero variance.

## Evaluation Results

### DGRO vs SFT Comparison

| Metric | SFT | DGRO | Diff |
|--------|-----|------|------|
| Recovery Rate | 99.98% | 99.6% | -0.38% |
| RR@5 | 95.0% | 79.4% | **-15.6%** |
| RR@10 | 99.4% | 98.0% | -1.4% |
| RR@15 | 99.9% | 99.2% | -0.7% |
| RR@20 | 99.98% | 99.6% | -0.38% |
| Avg Steps | 1.9 | 3.7 | +1.8 |
| **Diagnosis Accuracy** | 55.6% | **63.5%** | **+7.9%** |

### Key Observations

1. **Improved Diagnosis Accuracy (+7.9%)**
   - DGRO achieves 63.5% DA vs SFT's 55.6%
   - Suggests better understanding of root causes

2. **Reduced Early Efficiency (-15.6% RR@5)**
   - DGRO takes more steps to reach solutions
   - Avg steps increased from 1.9 to 3.7
   - May indicate more exploration behavior

3. **Near-Perfect Final Recovery**
   - Both models achieve ~99.6%+ final recovery
   - Long-term correctness preserved

## Interpretation

### Positive
- DGRO's higher diagnosis accuracy suggests it learned better debugging strategies
- The reward variance was successfully maintained (1.344 vs 0 in standard GRPO)
- Model learned more exploratory behavior

### Concerns
- Significant drop in RR@5 (early-step efficiency)
- More steps required on average
- Possible over-exploration due to high β_reward=2.0

## Model Artifacts

| Artifact | Path | Size |
|----------|------|------|
| DGRO LoRA Adapter | `/data/dgro_output/final/` | 174 MB |
| DGRO Merged Model | `/data/qwen3_dgro_merged/` | 16.4 GB |
| Evaluation Results | `outputs/experiments/2026-01-16/dgro_holdout_eval/` | - |
| Training Log | `/tmp/dgro_training.log` | - |

## Recommendations for Future Work

1. **Tune β_reward**: Try lower values (1.0, 0.5) to reduce over-exploration
2. **Curriculum Learning**: Focus on hard problems where SFT fails
3. **Combine with PRM**: Add process reward model for step-level guidance
4. **Hybrid Approach**: Early-stop DGRO exploration if solution found quickly

---

*Model: Qwen3-8B, Training Framework: TRL 0.24.0*
