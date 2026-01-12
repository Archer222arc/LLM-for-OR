# Bench → Eval → Training 连贯性设计

*返回: [项目计划](../PROJECT_PLAN.md)*

---

**核心问题**: Bench的设计决策如何指导Training？为什么这样设计Bench能让Training更有效？

---

## 连贯性1: Error Type → SFT数据多样性

```
Bench设计决策                    →    Training影响
─────────────────────────────────────────────────────
4种Error Type (A/B/C/D)          →    SFT数据覆盖4种诊断模式
                                       - Type D: 学习"直接冲突识别"
                                       - Type A: 学习"约束语义理解"
                                       - Type B/C: 学习"深度推理"

3层难度 (Easy/Medium/Hard)       →    课程学习 (Curriculum Learning)
                                       - 先用Easy数据warm-up
                                       - 逐步引入Hard数据
```

**实现指导**:
```python
def create_sft_curriculum(bench_data):
    """
    Bench的难度分层直接指导SFT的课程设计

    为什么这样设计Bench?
    - Easy (Type D): IIS直接给出答案 → 模型先学会"读IIS"
    - Medium (Type A): 需要理解约束语义 → 模型学会"理解约束"
    - Hard (Type B/C): 需要深度推理 → 模型学会"推理修复"
    """
    # Phase 1: Easy问题 (50% epochs)
    easy_data = [d for d in bench_data if d.difficulty == "easy"]

    # Phase 2: Easy + Medium (30% epochs)
    medium_data = [d for d in bench_data if d.difficulty in ["easy", "medium"]]

    # Phase 3: All difficulties (20% epochs)
    all_data = bench_data

    return [easy_data, medium_data, all_data]
```

---

## 连贯性2: IIS Ground Truth → Faithfulness Reward

```
Bench设计决策                    →    Training影响
─────────────────────────────────────────────────────
每个问题必须有IIS ground truth   →    GRPO可以计算Faithfulness penalty

IIS包含具体约束名称             →    可以精确匹配agent诊断 vs 实际IIS
                                       - 如果agent说"c1有问题"但IIS是{c2,c3}
                                       → Faithfulness penalty = -20
```

**为什么Bench必须保存IIS**:
```python
# Bench数据结构
class BenchProblem:
    original_model: str       # 原始可行模型
    sabotaged_model: str      # 破坏后的模型
    error_type: str           # A/B/C/D
    target: str               # 被破坏的约束/变量名
    iis: List[str]            # ⭐ IIS ground truth (关键!)
    ground_truth_fix: str     # 正确修复方法
    difficulty: str           # easy/medium/hard

# Training时的Reward计算
def compute_faithfulness_reward(agent_output, bench_problem):
    """
    没有Bench的IIS ground truth，这个reward无法计算!
    这就是Bench设计如何"指导"Training的核心
    """
    agent_diagnosis = parse_diagnosis(agent_output)  # agent说哪些约束有问题
    actual_iis = set(bench_problem.iis)              # Bench提供的ground truth

    overlap = agent_diagnosis & actual_iis
    if len(overlap) == 0:
        return -20  # 诊断完全错误
    return 0
```

---

## 连贯性3: Solver Status → Outcome Reward

```
Bench设计决策                    →    Training影响
─────────────────────────────────────────────────────
破坏后必须INFEASIBLE            →    GRPO有明确的outcome signal
修复后必须OPTIMAL               →    Outcome reward = +100/-50
四重验证Pipeline                →    训练数据100%可靠
```

**Bench验证 → Training可靠性**:
```python
# Bench的四重验证
def validate_bench_problem(problem):
    assert solve(problem.original).status == OPTIMAL      # 原模型可解
    assert solve(problem.sabotaged).status == INFEASIBLE  # 破坏后不可解
    assert problem.target in compute_iis(problem.sabotaged)  # IIS包含target
    assert solve(apply_fix(problem)).status == OPTIMAL    # 修复后可解

# 这确保了Training时:
# 1. Outcome reward有明确定义 (OPTIMAL=+100, else=-50)
# 2. 不会有"假阳性"问题 (solver说OPTIMAL但实际不是)
# 3. Ground truth fix是可验证的
```

---

## 连贯性4: NL描述 → SFT格式

```
Bench设计决策                    →    Training影响
─────────────────────────────────────────────────────
每个问题包含NL描述              →    SFT input包含problem context
约束有语义命名                  →    模型可以关联NL和约束名
                                       "demand_A" ↔ "需求约束A"
```

**SFT数据格式由Bench结构决定**:
```python
def bench_to_sft(bench_problem, trajectory):
    """
    Bench的数据结构直接决定了SFT的input格式

    如果Bench没有NL描述 → SFT只能学pattern matching
    如果Bench有NL描述   → SFT可以学semantic reasoning
    """
    return {
        "instruction": "Debug the infeasible optimization model.",
        "input": f"""
## Problem Description (来自Bench)
{bench_problem.nl_description}

## Model Code (来自Bench)
{bench_problem.sabotaged_model}

## Solver Feedback
Status: INFEASIBLE
IIS: {bench_problem.iis}
""",
        "output": f"""
<think>
分析IIS: {trajectory.iis_analysis}
关联NL描述: {trajectory.nl_reasoning}  # 只有Bench有NL才能学这个!
确定根因: {trajectory.root_cause}
</think>

Diagnosis: {trajectory.diagnosis}
Action: {trajectory.action}
"""
    }
```

---

## 总结: Bench设计原则 → Training效果

| Bench设计决策 | 为什么这样设计 | Training如何受益 |
|--------------|---------------|-----------------|
| 4种Error Type | 覆盖不同诊断难度 | SFT学到多样化pattern |
| 3层难度分级 | 基于IIS大小 | 支持课程学习 |
| IIS ground truth | Solver提供确定性答案 | Faithfulness reward可计算 |
| 四重验证 | 确保数据质量 | Training信号可靠 |
| NL描述 | 关联语义和约束 | 模型学会semantic reasoning |
| 约束命名规范 | 可读性+可匹配性 | 诊断可以精确评估 |

---

*相关文档*:
- [MDP设计](01_MDP_DESIGN.md) - MDP贯穿全流程
- [Bench构建](03_BENCH.md) - 数据构建详情
- [训练流程](05_TRAINING.md) - SFT/GRPO实现
