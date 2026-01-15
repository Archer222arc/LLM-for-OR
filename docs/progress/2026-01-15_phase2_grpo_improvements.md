# Phase 2: GRPO Improvement Implementation

**Date**: 2026-01-15
**Status**: Completed
**Prior**: [2026-01-15_randomized_naming_experiment.md](2026-01-15_randomized_naming_experiment.md)

---

## Objective

Implement four strategies to improve GRPO training based on 2025-2026 state-of-the-art techniques, addressing the reward variance collapse problem identified in Phase 1.

## Background

Phase 1 experiments showed:
- SFT achieves 95%+ RR@5 on all problem types
- GRPO cannot surpass SFT due to reward variance collapse
- When SFT solves 95% of problems, all GRPO generations get similar rewards → zero gradient

## Implementation Summary

### Strategy 1: Curriculum Learning on Failures (Phase 2.1)

**Files Created**:
| File | Description |
|------|-------------|
| `scripts/training/collect_sft_failures.py` | Extract SFT failure cases from evaluation database |
| `scripts/training/curriculum_grpo.py` | GRPO training focused on hard problems only |

**Key Features**:
- Extracts failures from SQLite evaluation results
- Classifies failure types: timeout, wrong_fix, wrong_diagnosis, partial_diagnosis
- Generates weakness profiles for targeted problem generation
- Trains GRPO only on problems where SFT fails (~5%)

### Strategy 2: Process Reward Model (Phase 2.2)

**Files Created/Modified**:
| File | Description |
|------|-------------|
| `src/training/process_reward_model.py` | ThinkPRM-style step-level reward model |
| `src/training/gurobi_rewards.py` | Modified to integrate PRM scoring |

**Key Features**:
- Auto-generates step-level labels using solver feedback (no manual annotation)
- Label rules:
  - `|IIS_t+1| < |IIS_t|` → label = 1.0 (IIS reduced)
  - `status == OPTIMAL` → label = 1.0 (solved)
  - `diagnosis ∩ actual_IIS ≠ ∅` → label = 0.5 (correct diagnosis)
- Lightweight PRM classifier (Qwen3-1.5B backbone)
- Integrated into composite reward function

### Strategy 3: DGRO Variance-Aware Training (Phase 2.3)

**Files Created**:
| File | Description |
|------|-------------|
| `scripts/training/train_dgro.py` | DGRO training with decoupled coefficients |

**Key Features**:
- Decoupled β_kl (KL penalty) and β_reward (reward scaling)
- Variance-aware advantage scaling: `advantage = (reward - baseline) * β_reward / (std + ε)`
- Optional Kalman filter baseline (KRPO)
- Handles low-variance scenarios with exploration noise
- Adaptive β scheduling option

### Strategy 4: Self-Play Bug Injection (Phase 2.4)

**Files Created**:
| File | Description |
|------|-------------|
| `scripts/training/self_play_training.py` | Iterative self-play training loop |

**Key Features**:
- Iterative loop: evaluate → identify weaknesses → generate problems → train
- WeaknessProfile class for systematic weakness analysis
- Integration with Saboteur for targeted problem generation
- Tracks progress across multiple rounds

---

## Technical Details

### DGRO Configuration (Recommended)

| Parameter | Standard GRPO | DGRO | Rationale |
|-----------|---------------|------|-----------|
| β_kl | 0.01 | **0.001** | Lower KL penalty for more exploration |
| β_reward | 1.0 | **2.0** | Amplify reward signal |
| temperature | 0.7 | **1.0** | Increase generation diversity |
| num_generations | 4 | **8** | More samples for variance estimation |

### PRM Label Distribution (Expected)

| Label | Meaning | Expected % |
|-------|---------|------------|
| 1.0 | Problem solved or IIS reduced | 20-30% |
| 0.5 | Correct diagnosis | 30-40% |
| 0.2 | Diagnostic action | 10-20% |
| 0.0 | No progress | 20-30% |

---

## Usage Workflow

```bash
# Phase 2.1: Collect SFT failures
python scripts/training/collect_sft_failures.py \
    --db outputs/experiments/2026-01-15/sft_holdout_eval/results.db \
    --model sft \
    --output data/training/sft_failures.json

# Phase 2.2: Generate PRM labels and train
python -m src.training.process_reward_model generate \
    --db outputs/experiments/2026-01-15/sft_holdout_eval/results.db \
    --output data/training/prm_labels.json

python -m src.training.process_reward_model train \
    --labels data/training/prm_labels.json \
    --output /data/prm_output

# Phase 2.3: DGRO training
python scripts/training/train_dgro.py \
    --model /data/qwen3_or_debug_merged \
    --failures data/training/sft_failures.json \
    --benchmark data/benchmarks/or_debug_bench_holdout \
    --output /data/dgro_output \
    --prm-path /data/prm_output \
    --beta-kl 0.001 --beta-reward 2.0

# Phase 2.4: Self-play loop
python scripts/training/self_play_training.py \
    --model /data/qwen3_or_debug_merged \
    --benchmark data/benchmarks/or_debug_bench_holdout \
    --output /data/self_play_output \
    --rounds 5
```

---

## Files Summary

| File | Lines | Description |
|------|-------|-------------|
| `scripts/training/collect_sft_failures.py` | 373 | SFT failure extraction |
| `scripts/training/curriculum_grpo.py` | 448 | Curriculum GRPO training |
| `src/training/process_reward_model.py` | 520 | Process Reward Model |
| `scripts/training/train_dgro.py` | 420 | DGRO training script |
| `scripts/training/self_play_training.py` | 480 | Self-play training loop |
| `src/training/gurobi_rewards.py` | +50 | PRM integration modifications |

**Total new code**: ~2,300 lines

---

## Research References

| Technique | Source | Application |
|-----------|--------|-------------|
| Curriculum Learning | E2H Reasoner (arXiv 2506.06632) | Focus on hard problems |
| ThinkPRM | OpenReview 2025 | Auto-label with solver feedback |
| DGRO | arXiv 2505.12951 | Variance-aware advantages |
| KRPO | arXiv 2025 | Kalman filter baseline |
| SWE-RL | arXiv 2512.18552 | Self-play injection loop |

---

## Next Steps

1. **VM Execution**: Run the complete pipeline on GPU server
2. **Evaluation**: Compare DGRO vs standard GRPO vs SFT
3. **Analysis**: Track reward variance throughout training
4. **Iteration**: Multiple self-play rounds if needed

---

*Implementation completed: 2026-01-15*
