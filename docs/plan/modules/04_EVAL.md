# Phase 2: 评估框架

*返回: [项目计划](../PROJECT_PLAN.md)*

---

## 2.1 指标实现 (核心代码)

### 任务2.1.1: Optimality Preservation (OP)
```python
def compute_op(recovered_obj, original_obj):
    """惩罚过度relaxation的修复"""
    if original_obj == 0:
        return 1.0 if recovered_obj == 0 else 0.0
    return max(0, 1 - abs(recovered_obj - original_obj) / abs(original_obj))
```

### 任务2.1.2: Feasibility Preservation (FP)
```python
def compute_fp(recovered_constrs: set, original_constrs: set) -> float:
    """惩罚删除过多约束的修复"""
    if not original_constrs:
        return 1.0
    return len(recovered_constrs & original_constrs) / len(original_constrs)
```

### 任务2.1.3: Faithfulness Score
```python
def compute_faithfulness(agent_diagnosis: set, actual_iis: set) -> float:
    """
    衡量agent诊断与实际IIS的一致性
    - 1.0: 完全准确
    - 0.0: 完全无关
    """
    if not agent_diagnosis or not actual_iis:
        return 0.0
    intersection = agent_diagnosis & actual_iis
    return len(intersection) / len(actual_iis)
```

---

## 2.2 Test-Time Compute实验设计

### 实验2.2.1: Reasoning Model vs Standard Model
```yaml
experiment:
  name: "test_time_compute_analysis"
  models:
    reasoning: [o1, o4-mini, DeepSeek-R1]
    standard: [gpt-4.1, gpt-5-mini, Qwen3-8B]
  metrics:
    - RR@5, RR@10, RR@20  # 不同compute budget
    - avg_steps_to_recovery
    - thinking_token_count (for reasoning models)
  analysis:
    - Plot: RR vs steps (是否diminishing returns?)
    - Compare: reasoning内置thinking vs 多步交互
```

**预期发现**:
- Reasoning models在Hard问题上优势明显
- Standard models在Easy问题上更高效

---

## 2.3 Few-shot Prompt设计

```python
FEW_SHOT_EXAMPLES = [
    {
        "problem": "Transportation LP with demand > supply",
        "iis": ["demand_A", "supply_total"],
        "diagnosis": "Demand constraint too tight",
        "action": "RELAX_CONSTRAINT demand_A 10%",
        "result": "OPTIMAL"
    },
    {
        "problem": "Assignment problem with infeasible matching",
        "iis": ["assign_1", "assign_2", "capacity"],
        "diagnosis": "Capacity constraint conflicts with assignments",
        "action": "DROP_CONSTRAINT capacity",
        "result": "OPTIMAL"
    }
]
```

---

## 论文实验设计

### Table 1: Main Results (Zero-shot)

| Model | Type | RR@5 | RR@10 | RR@20 | DA | TE |
|-------|------|------|-------|-------|----|----|
| GPT-5.2 | General | - | - | - | - | - |
| o4-mini | Reasoning | - | - | - | - | - |
| DeepSeek-R1 | Reasoning | - | - | - | - | - |
| Qwen3-8B | Base | - | - | - | - | - |
| **Qwen3-8B-SFT** | Ours | - | - | - | - | - |
| **Qwen3-8B-GRPO** | Ours | - | - | - | - | - |

**预期**: GRPO > SFT > Base，且接近o4-mini

### Table 2: Error Type Breakdown

| Model | Type A | Type B | Type C | Type D |
|-------|--------|--------|--------|--------|
| (显示不同错误类型的难度差异) |

**预期**: Type D最简单，Type B/C最难

### Table 3: Ablation - Reward Design

| Reward Config | RR@10 | DA |
|---------------|-------|----|
| Outcome only | - | - |
| + Process | - | - |
| + Faithfulness | - | - |

**预期**: Faithfulness penalty对DA提升显著

### Figure 1: Test-Time Compute Analysis
- X: Steps allowed (5, 10, 15, 20)
- Y: Recovery Rate
- Lines: Different model types
- **预期发现**: Reasoning models benefit more from extra steps

### Figure 2: GRPO Learning Curve
- X: Training steps
- Y: RR@10 on validation
- **预期**: 单调上升，~3000 steps收敛

---

*相关文档*:
- [Novelty定位](02_NOVELTY.md) - 指标与Novelty的映射
- [MDP设计](01_MDP_DESIGN.md) - 指标与MDP组件的对应
