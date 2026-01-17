# Phase 4 Curriculum Training - VM Automation Prompt

## Context

你正在Azure GPU VM (2×A100, 80GB VRAM) 上继续执行 OR-Debug-Bench 的 Phase 4 Curriculum Training。

**前置工作已完成**:
- ✅ Ablation实验完成，reward variance 1.14 → 18.3 (16x improvement)
- ✅ 最佳配置: `run_2_asym_clip` (mean reward +54.3)
- ✅ Code files: train_grpo_ablation.py, curriculum_scheduler.py, composite_reward.py

**目标**: 使用asym_clip配置运行4-stage curriculum training

---

## Task 1: Verify Environment

### 1.1 Pull Latest Code
```bash
cd /home/azureuser/LLM-for-OR
git pull origin main
```

### 1.2 Check Available Benchmarks
```bash
# 查看已生成的benchmarks
ls -la data/benchmarks/per_type/

# 预期: type_B_*, type_D_* 完整 (各200问题)
# 注意: type_A_*, type_C_* 可能不完整
```

### 1.3 Verify Ablation Results
```bash
# 确认最佳模型存在
ls -la /data/ablation_output/run_2_asym_clip/

# 查看训练日志
tail -50 /data/ablation_output/run_2_asym_clip/training.log
```

---

## Task 2: Run Curriculum Training

### 2.1 Start Curriculum Training
使用asym_clip配置运行4-stage curriculum:

```bash
cd /home/azureuser/LLM-for-OR

python scripts/training/curriculum_scheduler.py \
    --model /data/qwen3_or_debug_merged \
    --benchmarks_dir data/benchmarks/per_type \
    --output /data/curriculum_output \
    --best_config run_2_asym_clip \
    --num_stages 4 \
    --epochs_per_stage 2 \
    --use_composite_reward \
    2>&1 | tee logs/curriculum_training_$(date +%Y%m%d_%H%M%S).log
```

**预期运行时间**: ~4小时 (4 stages × 2 epochs)

### 2.2 Monitor Training Progress
在另一个terminal监控:

```bash
# 查看GPU使用
watch -n 5 nvidia-smi

# 查看训练日志
tail -f /data/curriculum_output/training.log

# 查看各stage进度
ls -la /data/curriculum_output/stage_*/
```

### 2.3 Curriculum Stages详解

| Stage | Epochs | Difficulty Mix | Expected Behavior |
|-------|--------|----------------|-------------------|
| 1 | 1-2 | Easy only | Learn basic patterns, RR@5 ≥ 95% |
| 2 | 3-4 | Easy + Medium | Multi-step reasoning, Medium RR@5 ≥ 80% |
| 3 | 5-6 | Medium + Hard | Complex diagnosis, Hard RR@5 ≥ 60% |
| 4 | 7-8 | Hard only | Expert-level, consolidate gains |

---

## Task 3: Intermediate Checkpoints

### 3.1 Save Stage Checkpoints
每个stage完成后自动保存checkpoint:

```
/data/curriculum_output/
├── stage_1/          # Easy只训练
│   ├── checkpoint/   # 模型checkpoint
│   └── metrics.json  # Stage 1指标
├── stage_2/          # Easy + Medium
├── stage_3/          # Medium + Hard
├── stage_4/          # Hard only
└── final/            # 最终合并模型
```

### 3.2 Validate Each Stage (可选)
如果想验证中间结果:

```bash
# 在Stage 2完成后快速验证
python scripts/evaluation/evaluate_local_model.py \
    --model /data/curriculum_output/stage_2/checkpoint \
    --dataset data/benchmarks/per_type/type_D_medium/dataset.json \
    --max_samples 20 \
    --output outputs/experiments/curriculum_validation/stage_2
```

---

## Task 4: Final Evaluation

### 4.1 Merge Final Model
训练完成后合并LoRA adapter:

```bash
python scripts/training/merge_grpo_adapter.py \
    --base_model /data/qwen3_or_debug_merged \
    --adapter_path /data/curriculum_output/final \
    --output /data/qwen3_or_debug_curriculum_merged
```

### 4.2 Comprehensive Evaluation
在所有benchmarks上评估:

```bash
# 启动SGLang服务器
source /data/envs/sglang/bin/activate
python -m sglang.launch_server \
    --model-path /data/qwen3_or_debug_curriculum_merged \
    --tensor-parallel-size 2 \
    --port 30000 &

sleep 60  # 等待服务器启动

# 运行评估
python scripts/evaluation/evaluate_llm.py \
    --model sglang \
    --sglang_url http://localhost:30000 \
    --dataset data/benchmarks/per_type \
    --output outputs/experiments/phase4_curriculum_eval \
    --all_benchmarks \
    --max_samples 50
```

### 4.3 Compare with Baselines
```bash
python scripts/evaluation/compare_models.py \
    --models "SFT:/data/qwen3_or_debug_merged" \
             "Ablation:/data/ablation_output/run_2_asym_clip" \
             "Curriculum:/data/qwen3_or_debug_curriculum_merged" \
    --benchmarks data/benchmarks/per_type \
    --output outputs/experiments/phase4_comparison
```

---

## Task 5: Generate Paper Figures

### 5.1 Learning Curves
```bash
python scripts/visualization/plot_phase4_results.py \
    --curriculum_log /data/curriculum_output/training.log \
    --ablation_logs /data/ablation_output/run_*/training.log \
    --output outputs/experiments/phase4_curriculum_eval/figures
```

### 5.2 Performance Comparison
```bash
python scripts/visualization/plot_phase4_results.py \
    --eval_results outputs/experiments/phase4_curriculum_eval \
    --comparison outputs/experiments/phase4_comparison \
    --output outputs/experiments/phase4_curriculum_eval/figures \
    --paper_figures
```

---

## Success Criteria

| Metric | SFT Baseline | Ablation (asym_clip) | Target Curriculum |
|--------|--------------|----------------------|-------------------|
| Easy RR@5 | ~90% | - | ≥95% |
| Medium RR@5 | ~70% | - | ≥85% |
| **Hard RR@5** | ~40% | - | **≥60%** |
| Mean Reward | - | +54.3 | **≥70** |
| DA | 55% | - | **≥70%** |

---

## Troubleshooting

### GPU OOM
如果遇到显存不足:
```bash
# 减少batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8
```

### Training Instability
如果loss震荡:
```bash
# 降低学习率
--learning_rate 5e-6

# 增加warmup
--warmup_ratio 0.1
```

### 训练中断恢复
```bash
python scripts/training/curriculum_scheduler.py \
    --resume_from /data/curriculum_output/stage_2 \
    ...
```

---

## Files to Update After Completion

1. `docs/progress/2026-01-17_phase4_progress.md` - 添加curriculum结果
2. Push结果到GitHub:
```bash
cd /home/azureuser/LLM-for-OR
git add docs/progress/ outputs/experiments/phase4_*/
git commit -m "feat: add Phase 4 curriculum training results"
git push origin main
```

---

*请按顺序执行Task 1-5。预计总时间: ~5小时 (训练4h + 评估1h)*
*如遇到错误，提供完整的error traceback。*
