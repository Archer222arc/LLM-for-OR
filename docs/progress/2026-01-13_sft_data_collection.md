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

## Training Environment Setup (Updated)

### Deep Research Results
- Framework: **LLaMA-Factory** (Qwen3原生支持, ACL 2024)
- LoRA配置: rank=32, alpha=64, target=all (7层)
- DeepSpeed: **ZeRO-2** (不是ZeRO-3，避免LoRA梯度问题)
- 训练时间: 25-40分钟 (2×A800)

详见: `docs/research/sft_research.md`

### Created Files

| File | Description |
|------|-------------|
| `scripts/training/convert_to_qwen_chat.py` | Alpaca → Qwen Chat格式转换 |
| `configs/training/qwen3_or_debug_lora.yaml` | LLaMA-Factory训练配置 |
| `configs/training/ds_config_zero2.json` | DeepSpeed ZeRO-2配置 |
| `configs/training/dataset_info.json` | 数据集注册 |

### Converted Data

| File | Samples | Description |
|------|---------|-------------|
| `sft_qwen_chat.train.jsonl` | 696 | GPT-5.2-chat训练集 |
| `sft_qwen_chat.val.jsonl` | 78 | 验证集 |
| `sft_combined.jsonl` | 5,216 | 全部数据 (GPT + Heuristic) |

### Training Command (VM)
```bash
# Install dependencies
pip install llamafactory deepspeed

# Train on 2×A800
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train configs/training/qwen3_or_debug_lora.yaml
```

---

## Next Steps

1. Transfer to VM with A800 GPUs
2. Run SFT training (~30 minutes)
3. Evaluate fine-tuned model on OR-Debug-Bench
4. Proceed to GRPO/RLVR training

---

*Related: [05_TRAINING.md](../plan/modules/05_TRAINING.md), [sft_research.md](../research/sft_research.md)*
