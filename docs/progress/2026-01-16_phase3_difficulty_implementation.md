# Phase 3+ Difficulty-Stratified Implementation

**Date**: 2026-01-16
**Status**: Completed
**Prior**: [2026-01-16_dgro_training_results.md](2026-01-16_dgro_training_results.md)

---

## Objective

Implement difficulty-stratified benchmarks and balanced DGRO training to address the over-exploration problem observed in Phase 2 DGRO training.

## Background

Phase 2 DGRO Results:
- DA: +7.9% (55.6% → 63.5%) ✓
- RR@5: -15.6% (95.0% → 79.4%) ✗ over-exploration

Root cause: Problems too easy (SFT achieves 95%+ RR@5), no room for RL improvement.

## Implementation Completed

### 1. Difficulty Generator Module (NEW)
**File**: `src/data_generation/difficulty_generator.py`

Created difficulty-stratified problem generation module with:
- `DifficultyLevel` enum (EASY, MEDIUM, DIFFICULT)
- `DifficultyConfig` dataclass with iis_range, error_types, expected_rr5

| Level | IIS Range | Error Types | Expected SFT RR@5 |
|-------|-----------|-------------|-------------------|
| Easy | 1-2 | A, B, C | ≥90% |
| Medium | 3-5 | D, E, F | 75-85% |
| Difficult | 5-10 | E, G, H, I | 50-70% |

### 2. Dataset Generator Update
**File**: `scripts/data_generation/generate_dataset.py`

Added `--difficulty` CLI argument:
```bash
python scripts/data_generation/generate_dataset.py \
    --difficulty difficult --n_problems 500 \
    --output data/benchmarks/or_debug_bench_difficult --seed 2026
```

### 3. Efficiency Reward Function (NEW)
**File**: `src/training/gurobi_rewards.py`

Added efficiency reward to prevent over-exploration:
```python
def compute_efficiency_reward(steps: int) -> float:
    if steps <= 2:
        return 10.0   # Quick solve
    elif steps <= 5:
        return 5.0    # Reasonable effort
    else:
        return -2.0 * (steps - 5)  # Over-exploration penalty
```

Functions added:
- `set_use_efficiency_reward(enabled: bool)`
- `get_use_efficiency_reward() -> bool`
- `compute_efficiency_reward(steps: int) -> float`
- `efficiency_reward(completions, step_counts, **kwargs) -> List[float]`

### 4. DGRO Training Script Update
**File**: `scripts/training/train_dgro.py`

Changes:
- Changed default `--beta-reward` from 2.0 to 0.5
- Added `--use-efficiency-reward` flag
- Integrated efficiency reward into training metadata

Usage:
```bash
python scripts/training/train_dgro.py \
    --model /data/qwen3_or_debug_merged \
    --dataset data/benchmarks/or_debug_bench_difficult/dataset.json \
    --output /data/dgro_difficult_output \
    --beta-reward 0.5 \
    --use-efficiency-reward \
    --num-epochs 3
```

## Execution Results

### Benchmarks Generated

| Benchmark | Problems | Error Types | IIS Range |
|-----------|----------|-------------|-----------|
| or_debug_bench_easy | 438 | A, B, C | avg=4.3 |
| or_debug_bench_medium | 500 | D, E, F | avg=3.4 |
| or_debug_bench_difficult | 500 | E, G, H, I | avg=1.9 |

### SFT Evaluation Results

| Difficulty | RR    | RR@5  | RR@10 | Avg Steps | DA    |
|------------|-------|-------|-------|-----------|-------|
| **Easy**   | 100.0% | 87.0% | 98.0% | 2.73 | 64.6% |
| **Medium** | 100.0% | 97.0% | 100.0% | 1.74 | 31.4% |
| **Difficult** | 100.0% | 90.0% | 100.0% | 2.69 | 79.4% |

### Key Finding: Difficulty Gradient NOT Observed

**Expected**: Easy ≥90%, Medium ≤85%, Difficult ≤70% (RR@5)
**Observed**: Easy 87%, Medium 97%, Difficult 90% (RR@5)

**Analysis**:
1. IIS sizes didn't vary as planned (difficult avg IIS was 1.9, not 5-10)
2. MDP-advantage types (G/H/I) may actually be easier for the trained model
3. Error type difficulty assumptions may be incorrect
4. SFT model already achieves high performance, leaving little room for RL

### Implications

- Training DGRO on these benchmarks may not show significant improvement
- Need to revise difficulty definitions based on actual error type hardness
- Consider filtering by diagnosis accuracy (Medium DA=31.4% suggests harder diagnosis)

## Next Steps

1. **Option A**: Proceed with DGRO training as planned (with efficiency reward)
2. **Option B**: Revise difficulty definitions based on empirical results
3. **Option C**: Focus on problems where DA is low (Medium tier)

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/data_generation/difficulty_generator.py` | CREATE | Difficulty stratification module |
| `scripts/data_generation/generate_dataset.py` | MODIFY | Added --difficulty flag |
| `src/training/gurobi_rewards.py` | MODIFY | Added efficiency reward |
| `scripts/training/train_dgro.py` | MODIFY | Added --use-efficiency-reward, β=0.5 |

---

*Implementation: Qwen3-8B, Training Framework: TRL 0.24.0*
