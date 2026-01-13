# Progress: SFT Data Collection

**Date**: 2026-01-13
**Status**: Completed
**Phase**: Phase 3 - Training Data Collection

---

## Summary

Completed SFT (Supervised Fine-Tuning) data collection from gpt-5.2-chat teacher model for Qwen3-8B fine-tuning.

---

## Completed Tasks

### 1. Parallel Collection Script Enhancement
- Added `--workers N` for parallel collection (ThreadPoolExecutor)
- Changed default behavior to **incremental** (auto-resume if output exists)
- Added `--overwrite` flag for explicit fresh start
- Implemented thread-safe JSON saving with `fcntl` file locking

### 2. GPT-5.2-Chat Data Collection
| Metric | Value |
|--------|-------|
| Total Problems | 1000 |
| Successful | 774 (77.4%) |
| Failed | 226 |
| Workers | 4 |

### 3. SFT Data Format
```json
{
  "instruction": "Debug the infeasible optimization model...",
  "input": "## Problem\nID: ...\n## IIS\nConflicting Constraints: [...]\n## Model Structure\n...",
  "output": "<think>\nStep N: reasoning about IIS...\n</think>\n\nAction: RELAX_CONSTRAINT(...)",
  "metadata": {
    "problem_id": "mip_typeA_000",
    "error_type": "A",
    "difficulty": "medium",
    "steps": 4,
    "agent": "SFT-Teacher-gpt-5.2-chat"
  }
}
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/training/sft_gpt52chat.json` | 774 SFT samples from gpt-5.2-chat |
| `data/training/sft_heuristic.json` | 4,442 samples from HeuristicAgent (baseline) |

---

## Script Updates

### `scripts/training/collect_sft_data.py`
- Thread-local agent storage for parallel execution
- Incremental JSON saving with file locking
- Auto-resume: detects existing output and skips completed problems
- Progress reporting every 20 problems

Key functions:
- `collect_trajectory()`: Run agent, extract successful trajectory
- `format_state_for_sft()`: Convert state to input format
- `format_action_for_sft()`: Convert action to output with `<think>` tags
- `save_result_incremental()`: Thread-safe append to JSON

---

## Next Steps

1. Research SFT training framework (LLaMA-Factory, etc.)
2. Convert data format for chosen framework
3. Fine-tune Qwen3-8B with LoRA on A800

---

*Related: [05_TRAINING.md](../plan/modules/05_TRAINING.md)*
