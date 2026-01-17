# Phase 4 VM Automation Prompt for Claude Code

## Context

你正在Azure GPU VM (2×A100, 80GB VRAM) 上继续执行 OR-Debug-Bench 的 Phase 4: RL Training Improvement。

**目标**: 让RL训练(GRPO/DGRO)超过SFT baseline，验证MDP框架的重要性。

**前置工作已完成**:
- Phase 4.1核心代码已实现并push到GitHub
- Per-type difficulty configuration (4 types × 3 difficulties = 12 configs)
- generate_dataset.py 支持 `--generate_all_per_type` 批量生成

## Task 1: Setup & Generate Benchmarks

### 1.1 Pull Latest Code
```bash
cd /home/azureuser/LLM-for-OR
git pull origin main
```

### 1.2 Generate 12 Per-Type Benchmarks
生成每种类型(A/B/C/D) × 难度(easy/medium/hard) 的benchmark，每个200个问题：

```bash
python scripts/data_generation/generate_dataset.py \
    --generate_all_per_type \
    --n_problems 200 \
    --output data/benchmarks/per_type \
    --seed 2026
```

预期输出：12个目录，共2400个问题
```
data/benchmarks/per_type/
├── type_A_easy/    (200 problems)
├── type_A_medium/  (200 problems)
├── type_A_hard/    (200 problems)
├── type_B_easy/    (200 problems)
...
└── type_D_hard/    (200 problems)
```

### 1.3 Validate Difficulty Gradient
用SFT模型评估各benchmark的RR@5，验证difficulty gradient：

```bash
# 对每个benchmark运行评估
for benchmark in data/benchmarks/per_type/type_*; do
    python scripts/evaluation/evaluate_llm.py \
        --model /data/qwen3_or_debug_merged \
        --dataset $benchmark/dataset.json \
        --output outputs/experiments/phase4_difficulty_validation/$(basename $benchmark) \
        --max_samples 50 \
        --max_turns 10
done
```

**预期结果**:
| Difficulty | Target SFT RR@5 |
|------------|-----------------|
| easy | ≥90% |
| medium | 60-80% |
| hard | 30-50% |

## Task 2: Implement DAPO Ablation Trainer

### 2.1 Create train_grpo_ablation.py
创建 `scripts/training/train_grpo_ablation.py`，实现DAPO风格的GRPO修改：

**关键配置**:
```python
ablation_configs = {
    "run_0_baseline": {"beta": 0.001, "clip_range": 0.2},
    "run_1_no_kl": {"beta": 0.0, "clip_range": 0.2},
    "run_2_asym_clip": {"beta": 0.0, "clip_range": 0.2, "clip_range_high": 0.28},
    "run_3_dynamic": {"beta": 0.0, "clip_range": 0.2, "clip_range_high": 0.28, "dynamic_sampling": True},
    "run_4_token_loss": {"beta": 0.0, "clip_range": 0.2, "clip_range_high": 0.28, "dynamic_sampling": True, "token_loss": True},
}
```

### 2.2 Run 5 Ablation Experiments on Type D Medium
```bash
# 在Type D medium上运行ablation（有足够的改进空间）
for run in run_0_baseline run_1_no_kl run_2_asym_clip run_3_dynamic run_4_token_loss; do
    python scripts/training/train_grpo_ablation.py \
        --model /data/qwen3_or_debug_merged \
        --dataset data/benchmarks/per_type/type_D_medium/dataset.json \
        --output /data/ablation_output/$run \
        --config $run \
        --num_epochs 2
done
```

### 2.3 Analyze Ablation Results
比较各配置的：
- Reward variance (target: > 0.5)
- RR@5 improvement
- DA improvement
- Training stability

## Task 3: Implement Composite Reward

### 3.1 Create composite_reward.py
创建 `src/training/composite_reward.py`：

```python
def compute_composite_reward(trajectory, ground_truth_iis):
    # Outcome (0.5 weight)
    r_outcome = 100.0 if status == OPTIMAL else -50.0

    # Diagnosis (0.3 weight) - explicit DA signal
    pred_iis = extract_diagnosis(trajectory)
    da_score = len(pred_iis & ground_truth_iis) / len(ground_truth_iis)
    r_diagnosis = 20.0 * da_score

    # Efficiency (0.2 weight)
    r_efficiency = 10.0 if steps <= 2 else (5.0 if steps <= 5 else -2*(steps-5))

    return 0.5 * r_outcome + 0.3 * r_diagnosis + 0.2 * r_efficiency
```

## Task 4: Implement Curriculum Training

### 4.1 Create curriculum_scheduler.py
创建 `scripts/training/curriculum_scheduler.py`：

**4-Stage Curriculum**:
```
Stage 1 (Epochs 1-2): easy only
Stage 2 (Epochs 3-4): easy + medium
Stage 3 (Epochs 5-6): medium + hard
Stage 4 (Epochs 7-8): hard only
```

### 4.2 Run Curriculum Training
使用最佳ablation配置：

```bash
python scripts/training/curriculum_scheduler.py \
    --model /data/qwen3_or_debug_merged \
    --benchmarks_dir data/benchmarks/per_type \
    --output /data/curriculum_output \
    --best_config run_X_xxx  # 从ablation选择
    --num_stages 4 \
    --epochs_per_stage 2
```

## Task 5: Final Evaluation

### 5.1 Comprehensive Evaluation
```bash
python scripts/evaluation/evaluate_llm.py \
    --model /data/curriculum_output/final \
    --dataset data/benchmarks/per_type \
    --output outputs/experiments/phase4_final_eval \
    --all_benchmarks
```

### 5.2 Generate Paper Figures
```bash
python scripts/visualization/plot_phase4_results.py \
    --input outputs/experiments/phase4_final_eval \
    --output outputs/experiments/phase4_final_eval/figures
```

## Success Criteria

| Metric | SFT Baseline | Target GRPO+ |
|--------|--------------|--------------|
| Hard RR@5 | ~40% | **60%+** |
| Overall DA | 55% | **70%+** |
| Reward Variance | 0 | >0.5 |

## Files to Create

1. `scripts/training/train_grpo_ablation.py` - DAPO ablation trainer
2. `src/training/composite_reward.py` - Multi-objective reward
3. `scripts/training/curriculum_scheduler.py` - 4-stage curriculum
4. `scripts/visualization/plot_phase4_results.py` - Paper figures

## Notes

- 使用SGLang部署进行评估: `python -m sglang.launch_server --model-path /data/qwen3_or_debug_merged --port 30000`
- 训练使用TRL 0.24.0 GRPOTrainer
- 所有实验结果保存到 `outputs/experiments/phase4_*/`
- 进度日志更新到 `docs/progress/2026-01-17_phase4_*.md`

---

*请按顺序执行Task 1-5，每个Task完成后汇报进度。如遇到错误，提供完整的error traceback。*
