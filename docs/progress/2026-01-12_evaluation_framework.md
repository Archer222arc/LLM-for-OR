# 2026-01-12: 评估框架实现 (Phase 2)

**状态**: 完成

---

## 完成内容

### 1. 指标扩展 (`src/evaluation/metrics.py`)

新增核心指标:

| 指标 | 公式 | 用途 |
|------|------|------|
| **OP** (Optimality Preservation) | `1 - |obj_new - obj_orig| / |obj_orig|` | 惩罚过度relaxation |
| **FP** (Feasibility Preservation) | `|constrs_kept ∩ constrs_orig| / |constrs_orig|` | 惩罚删除过多约束 |
| **RR@k** | 在k步内恢复的比例 | Test-time compute分析 |
| **Faithfulness** | `|diagnosed ∩ actual_iis| / |actual_iis|` | 诊断与IIS一致性 |

新增方法:
- `compute_op()` - Optimality Preservation计算
- `compute_fp()` - Feasibility Preservation计算
- `compute_rr_at_k()` - Recovery Rate at k steps
- `compute_faithfulness()` - Faithfulness (DA别名)

`EpisodeResult` 新增字段:
- `original_objective`, `recovered_objective` - OP计算用
- `original_constraint_count`, `remaining_constraint_count` - FP计算用
- `original_constraints`, `remaining_constraints` - 约束名集合
- `success_at_step` - 成功时的步数 (RR@k用)

### 2. Few-shot支持 (`src/agents/prompts.py`)

新增常量:
- `FEW_SHOT_EXAMPLES` - 3个典型示例 (Transportation/Assignment/Production)

新增函数:
- `format_few_shot_prompt(n_examples)` - 格式化few-shot示例
- `get_system_prompt(use_few_shot, n_examples)` - 获取系统提示词

### 3. 评估脚本 (`scripts/evaluate_llm.py`)

**功能**:
- 加载数据集
- 创建多种agent (LLM/Baseline)
- 运行评估并计算所有指标
- 按Error Type和Difficulty分类统计
- 生成JSON结果报告
- 打印对比表格

**使用方式**:
```bash
# 评估所有模型
python scripts/evaluate_llm.py --config configs/experiments/eval_config.yaml

# 评估单个模型
python scripts/evaluate_llm.py --model gpt-4.1 --dataset data/benchmarks/or_debug_bench_full

# 快速测试
python scripts/evaluate_llm.py --limit 10
```

### 4. 配置文件 (`configs/experiments/eval_config.yaml`)

支持的模型配置:
- **Baselines**: HeuristicAgent, GreedyDropAgent, RandomAgent
- **GPT系列**: gpt-4.1, gpt-4.1-mini, gpt-5-mini
- **Reasoning**: o1, o4-mini
- **DeepSeek**: DeepSeek-R1, DeepSeek-V3.2

配置项:
- `evaluation.max_steps` - 最大步数
- `evaluation.n_episodes` - 每问题重复次数
- `output.save_trajectories` - 是否保存轨迹

---

## 产出统计

| 文件 | 操作 | 行数 |
|------|------|------|
| `src/evaluation/metrics.py` | 修改 | +130行 |
| `src/evaluation/episode_stats.py` | 修改 | +20行 |
| `src/evaluation/benchmark_runner.py` | 修改 | +10行 |
| `src/agents/prompts.py` | 修改 | +140行 |
| `src/agents/__init__.py` | 修改 | +5行 |
| `scripts/evaluate_llm.py` | **新建** | 390行 |
| `configs/experiments/eval_config.yaml` | **新建** | 90行 |

**总计**: ~785行新增/修改代码

---

## 验证结果

小规模测试 (5问题, 3 baselines):

| Agent | RR | RR@5 | RR@10 | OP | DA |
|-------|-----|------|-------|-----|-----|
| HeuristicAgent | 100% | 100% | 100% | 100% | 45% |
| GreedyDropAgent | 100% | 100% | 100% | 100% | 45% |
| RandomAgent | 100% | 80% | 100% | 76% | 45% |

**观察**:
- Heuristic/Greedy都在1步内解决 (RR@5=100%)
- RandomAgent需要更多步数 (avg 3步), RR@5=80%
- RandomAgent的OP较低 (76%) 因为随机删除约束

---

## 输出示例

```json
{
  "timestamp": "2026-01-12T01:05:47",
  "dataset": "data/benchmarks/or_debug_bench_test_v2",
  "n_problems": 5,
  "results": [
    {
      "agent_name": "HeuristicAgent",
      "recovery_rate": 1.0,
      "rr_at_5": 1.0,
      "rr_at_10": 1.0,
      "optimality_preservation": 1.0,
      "diagnosis_accuracy": 0.45,
      "by_error_type": {"A": {"n": 5, "recovery_rate": 1.0}},
      "by_difficulty": {"hard": {"n": 5, "recovery_rate": 1.0}}
    }
  ]
}
```

---

## 下一步

1. **全规模Baseline评估** - 在4462问题上运行baselines
2. **LLM评估** - GPT-4.1, o4-mini等模型评估
3. **Test-time Compute分析** - RR vs Steps曲线
4. **结果可视化** - 生成论文图表

---

*关联文档*:
- [04_EVAL.md](../plan/modules/04_EVAL.md) - Phase 2评估框架计划
- [2026-01-12_robust_injection.md](2026-01-12_robust_injection.md) - Phase 1数据构建
