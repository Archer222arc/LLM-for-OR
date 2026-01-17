# Phase 4: RL Training Improvement Plan

**Date**: 2026-01-17
**Status**: PLANNING
**Prior**: [2026-01-17_phase3.2_da_filtering_results.md](2026-01-17_phase3.2_da_filtering_results.md)

---

## Objective

Improve RL training (GRPO/DGRO) to demonstrate clear improvements over SFT baseline, validating the MDP framework's importance for OR-Debug-Bench (NeurIPS 2026).

## Background: Current Bottleneck

| Problem | Root Cause | Evidence |
|---------|------------|----------|
| Reward variance = 0 | SFT achieves 95%+ RR, all samples get same reward | GRPO training logs |
| DGRO over-exploration | High β_reward causes excessive steps | RR@5: 95%→79.4% |
| DA unchanged at 27% | No explicit diagnosis reward signal | Phase 3.2 results |
| Problems too easy | Single-layer IIS, simple error types | Medium DA=31.4% but RR@5=97% |

**Key Insight**: RL cannot improve when baseline already saturates the task. Need harder problems + better reward design.

---

## Research Foundation (2025-2026 SOTA)

### 1. DAPO (ByteDance, 2025)
**Source**: [arXiv:2503.14476](https://arxiv.org/html/2503.14476v1)

Key techniques:
- **No KL Divergence**: Not needed for RLVR (β=0 now TRL default)
- **Asymmetric Clipping**: [ε_low, ε_high] = [0.2, 0.28] for new token exploration
- **Dynamic Sampling**: Filter flat-reward batches
- **Token-Level Loss**: Better stability than sequence-level

### 2. Curriculum RL (E2H Reasoner, 2025)
**Source**: [arXiv:2506.06632](https://arxiv.org/abs/2506.06632)

Key insight: Direct training on hard tasks → sparse rewards → no learning. Progressive curriculum:
- Trivial → Easy → Medium → Hard
- Partial rewards for format correctness
- Builds foundational skills before complex tasks

### 3. RLVR with Composite Rewards
**Source**: [DeepSeek R1](https://arxiv.org/abs/2501.12948), [Sebastian Raschka Survey](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)

- Verifiable rewards from solvers (Gurobi as oracle)
- Multi-objective: Outcome + Process + Efficiency
- Dense feedback outperforms sparse signals

---

## Proposed Solution: 4-Component Framework

### Component 1: Per-Type Difficulty-Stratified Benchmark

**Approach**: Per-Type stratification - each error type (A/B/C/D) has its own easy/medium/hard variants.

**Per-Type Difficulty Definition**:

| Type | Easy | Medium | Hard |
|------|------|--------|------|
| **Type A** (Semantic) | 1 constraint | 2-3 constraints | 4+ constraints, obfuscated |
| **Type B** (Bounds) | 1 variable | 3 variables | 5+ variables, chained |
| **Type C** (Logic) | Single error | Nested errors | Multi-level chains |
| **Type D** (RHS) | Single flip | 2-3 conflicts | Cascading deps |

**Target SFT Performance**:

| Difficulty | Target SFT RR@5 | RL Improvement Target |
|------------|-----------------|----------------------|
| Easy | ≥90% | +3-5% |
| Medium | 60-80% | +10-15% |
| **Hard** | 30-50% | **+15-25%** |

### Component 2: DAPO-Style GRPO Modifications (Incremental Ablation)

**Approach**: Incremental ablation - add one modification at a time to measure individual impact.

| Run | Modification | Hypothesis |
|-----|--------------|------------|
| **Run 0** | Baseline GRPO | Establish baseline |
| **Run 1** | +KL Removal (β=0) | More exploration |
| **Run 2** | +Asymmetric Clipping | Better new token learning |
| **Run 3** | +Dynamic Sampling | Training efficiency |
| **Run 4** | +Token-Level Loss | Training stability |

### Component 3: 4-Stage Curriculum Training

```
Stage 1 (Epochs 1-2): Easy only
  → Learn basic diagnostic patterns
  → Expected: RR@5 ≥ 95%, DA ≥ 70%

Stage 2 (Epochs 3-4): Easy + Medium
  → Multi-step reasoning
  → Expected: Medium RR@5 ≥ 80%

Stage 3 (Epochs 5-6): Medium + Hard
  → Complex diagnosis
  → Expected: Hard RR@5 ≥ 60%

Stage 4 (Epochs 7-8): Hard + Expert
  → Expert-level debugging
  → Expected: Expert RR@5 ≥ 40%
```

### Component 4: Multi-Objective Composite Reward

```python
def compute_composite_reward(trajectory, ground_truth_iis):
    # Outcome (sparse, terminal)
    r_outcome = 100.0 if status == OPTIMAL else -50.0

    # Diagnosis (dense, explicit DA signal)
    pred_iis = extract_diagnosis(trajectory)
    da_score = |pred_iis ∩ ground_truth_iis| / |ground_truth_iis|
    r_diagnosis = 20.0 * da_score

    # Efficiency (step penalty)
    r_efficiency = 10.0 if steps <= 2 else (5.0 if steps <= 5 else -2*(steps-5))

    # Composite: [outcome, diagnosis, efficiency]
    return 0.5 * r_outcome + 0.3 * r_diagnosis + 0.2 * r_efficiency
```

---

## Implementation Plan

**Environment**: Azure GPU VM (2×A100, 80GB VRAM)

### Phase 4.1: Per-Type Difficulty Data Generation (2-3 days)
1. Update `src/data_generation/saboteur_agent.py` for per-type difficulty
2. Generate 12 benchmarks (4 types × 3 difficulties, 200 each = 2400 problems)
3. Validate difficulty gradient with SFT evaluation

### Phase 4.2: DAPO Ablation Implementation (3-4 days)
1. Create `scripts/training/train_grpo_ablation.py`
2. Run 5 ablation experiments on Type D medium
3. Analyze and select best configuration

### Phase 4.3: Composite Reward + Curriculum (3-4 days)
1. Implement `src/training/composite_reward.py`
2. Implement `scripts/training/curriculum_scheduler.py`
3. Run curriculum training with best DAPO config

### Phase 4.4: Evaluation & Paper Figures (2-3 days)
1. Per-type, per-difficulty RR@k curves
2. DAPO ablation table
3. Publication-ready figures

---

## Expected Outcomes

| Metric | SFT Baseline | Target (GRPO+) | Improvement |
|--------|--------------|----------------|-------------|
| Easy RR@5 | 87% | 92% | +5% |
| Medium RR@5 | 97% | 95% | -2% (acceptable) |
| **Hard RR@5** | ~60%* | **75%** | **+15%** |
| **Expert RR@5** | ~30%* | **50%** | **+20%** |
| Overall DA | 55% | **70%** | **+15%** |

*Estimated on new hard/expert benchmarks

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/data_generation/saboteur_agent.py` | MODIFY | Cascading error injection |
| `src/data_generation/difficulty_generator.py` | MODIFY | HARD/EXPERT levels |
| `scripts/training/train_grpo_dapo.py` | CREATE | DAPO-style trainer |
| `scripts/training/curriculum_scheduler.py` | CREATE | 4-stage curriculum |
| `src/training/composite_reward.py` | CREATE | Multi-objective reward |
| `data/benchmarks/or_debug_bench_hard/` | CREATE | Hard benchmark |
| `data/benchmarks/or_debug_bench_expert/` | CREATE | Expert benchmark |

---

## Verification Plan

1. **Unit Tests**: DAPO mods, composite reward, curriculum scheduler
2. **Integration**: Short training (100 steps), verify reward variance > 0.5
3. **E2E**: Full curriculum, compare vs SFT on held-out test

---

## References

- [DAPO: Open-Source LLM RL at Scale](https://arxiv.org/html/2503.14476v1)
- [E2H Curriculum RL](https://arxiv.org/abs/2506.06632)
- [GRPO++ Tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [DeepSeek R1](https://arxiv.org/abs/2501.12948)
- [State of LLM Reasoning Training](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)

---

*Plan Version: 1.0*
*Created: 2026-01-17*
