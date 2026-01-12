# 2026-01-11: 多模型LLM评估

**状态**: 完成

## 完成内容

### Phase 5: 并行测试框架

#### 1. `src/evaluation/benchmark_runner.py` 扩展
- `compare_agents_parallel()`: Agent级别并行评估
- `_evaluate_single_agent_worker()`: 多进程worker函数
- 使用ProcessPoolExecutor (Gurobi线程安全要求)

#### 2. `scripts/run_llm_experiment.py` 扩展
- `--parallel`: 启用并行模式
- `--max-workers N`: 控制并发数 (默认4)

### Phase 6: 多类型数据集生成

#### 3. `data/synthetic/debug_bench_v2/`
- 21个问题 (1 Type A + 20 Type D)
- Type B/C生成失败率高 (不保证infeasibility)

### Phase 6: 11模型完整评估

#### 4. 评估配置
- 配置: `configs/experiments/llm_eval_all_types.yaml`
- 数据集: `debug_bench_v2` (21问题)
- 并行: 6 workers

## 评估结果

| 模型 | Recovery Rate | Avg Steps | Step Efficiency |
|------|---------------|-----------|-----------------|
| gpt-5.2-chat | 100.0% | 1.24 | 157.08 |
| gpt-5-mini | 100.0% | 1.33 | 145.43 |
| o4-mini | 100.0% | 1.38 | 140.38 |
| DeepSeek-V3.2 | 100.0% | 1.52 | 127.59 |
| gpt-5-nano | 100.0% | 1.81 | 108.61 |
| o1 | 100.0% | 2.00 | 96.74 |
| DeepSeek-R1-0528 | 100.0% | 2.33 | 86.24 |
| gpt-4.1 | 100.0% | 2.38 | 83.30 |
| Llama-3.3-70B-Instruct | 100.0% | 3.14 | 62.33 |
| Kimi-K2-Thinking | 95.2% | 3.81 | 49.19 |
| **gpt-4.1-mini** | **66.7%** | **13.57** | **14.18** |

### 关键发现

1. **顶级模型**: gpt-5.2-chat, gpt-5-mini, o4-mini在1.4步内完成
2. **推理模型**: o1, o4-mini, DeepSeek-R1均100%恢复
3. **落后模型**: gpt-4.1-mini仅66.7%恢复率，平均13.6步
4. **Kimi-K2-Thinking**: 95.2%恢复率，有1个失败案例

## 验证结果

```bash
# 并行评估命令
source ~/miniforge3/etc/profile.d/conda.sh && conda activate llm4or && \
ALL_PROXY="" python scripts/run_llm_experiment.py \
  --config configs/experiments/llm_eval_all_types.yaml \
  --parallel --max-workers 6

# 输出
outputs/experiments/llm_eval_all_types/
├── report.md        # Markdown报告
├── results.json     # 结构化结果
└── trajectories/    # 详细轨迹
```

## 产出统计

| 文件 | 说明 |
|------|------|
| benchmark_runner.py扩展 | +50行并行逻辑 |
| run_llm_experiment.py扩展 | +20行CLI参数 |
| llm_eval_all_types.yaml | 新实验配置 |
| debug_bench_v2/ | 21问题数据集 |

## Phase 7: 诊断准确率(DA)指标实现 - ✅ 已完成

### 新增文件修改

| 文件 | 修改内容 |
|------|----------|
| `src/evaluation/metrics.py` | 添加`diagnosed_constraints`和`ground_truth_iis`字段到`EpisodeResult`，实现`compute_diagnosis_accuracy()`和`compute_diagnosis_precision()` |
| `src/evaluation/episode_stats.py` | 更新`EpisodeTracker`记录诊断的约束 |
| `src/evaluation/benchmark_runner.py` | 添加`ground_truth_iis`到`BenchmarkProblem`和数据传递 |
| `scripts/run_llm_experiment.py` | 更新报告生成器包含DA指标 |

### DA指标定义

```
Diagnosis Accuracy (DA) = |diagnosed ∩ actual_IIS| / |actual_IIS|
Diagnosis Precision = |diagnosed ∩ actual_IIS| / |diagnosed|
```

- **DA (Recall)**: 模型识别了多少实际IIS约束
- **Precision**: 模型诊断的约束中有多少是正确的

### 快速测试验证

```json
{
  "gpt-5.2-chat": {
    "diagnosis_accuracy": 0.25,
    "diagnosis_precision": 1.0
  },
  "gpt-4.1-mini": {
    "diagnosis_accuracy": 0.25,
    "diagnosis_precision": 1.0
  }
}
```

## 下一步

1. **分析gpt-4.1-mini失败案例** - 诊断错误 vs 修复错误
2. **修复Type A/B/C生成** - 扩展错误类型覆盖
3. **运行完整评估包含DA** - 11模型对比DA指标
