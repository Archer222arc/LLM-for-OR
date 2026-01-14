# SQLite Storage Refactor for Local Model Evaluation

**Date**: 2026-01-14
**Status**: Completed

---

## Overview

Refactored `scripts/evaluation/evaluate_local_model.py` to use SQLite database storage instead of JSON files, aligning with project conventions in CLAUDE.md.

---

## Changes Made

### 1. New Imports and Dependencies

```python
import subprocess
import yaml
from datetime import datetime
from src.evaluation.result_db import ResultDB
```

### 2. Added `setup_experiment_dir()` Function

Creates standardized experiment directory structure:
- `config.yaml`: Experiment configuration
- `git_hash.txt`: Git commit hash for reproducibility
- `results.db`: SQLite database (primary storage)

### 3. Modified `evaluate_model()` Function

**New Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `outputs/experiments` | Base output directory |
| `exp_name` | str | `{model}_{limit}samples` | Experiment name |

**Key Changes**:
- Database-based resume via `db.get_completed_problems()`
- Incremental saves via `db.save_episode_result()`
- Summary saved via `db.update_summary()`
- JSON export via `db.export_json()` for backward compatibility

---

## New CLI Usage

```bash
# Basic evaluation
python scripts/evaluation/evaluate_local_model.py \
  --model /data/qwen3_or_debug_merged \
  --limit 200

# SGLang with concurrency
python scripts/evaluation/evaluate_local_model.py \
  --backend sglang \
  --model /data/qwen3_or_debug_merged \
  --limit 200 \
  --concurrency 8 \
  --exp-name sft_sglang_200
```

---

## Output Structure

```
outputs/experiments/2026-01-14/{exp_name}/
├── config.yaml     # Experiment configuration
├── git_hash.txt    # Git commit hash
├── results.db      # SQLite database (primary storage)
└── results.json    # Exported JSON (backward compatibility)
```

---

## Verification Results

- Sequential evaluation: 2/2 saved to SQLite
- Resume: Correctly detected completed problems
- Concurrent evaluation: 10/10 saved (WAL mode thread-safe)

---

*Implemented following project specifications in `.claude/CLAUDE.md`*
