# MDP设计与各部分定位

*返回: [项目计划](../PROJECT_PLAN.md)*

---

**核心Novelty**: 从静态Pass@1评估 → MDP策略评估 π(aₜ | sₜ)

## MDP形式化定义

```
MDP = (S, A, T, R, γ)

State S = {
    problem_nl: str,        # 问题自然语言描述
    model_code: str,        # 当前模型代码
    solver_status: enum,    # OPTIMAL/INFEASIBLE/UNBOUNDED/ERROR
    iis_log: List[str],     # IIS约束列表
    history: List[Action],  # 历史动作序列
    step: int               # 当前步数
}

Action A = {
    # Diagnosis Actions (信息获取)
    GET_IIS(),              # 获取IIS
    CHECK_SLACK(constr),    # 检查约束松弛量
    EXPLAIN_CONFLICT(),     # 解释冲突原因

    # Repair Actions (模型修改)
    RELAX_CONSTRAINT(constr, delta),  # 放松约束RHS
    DROP_CONSTRAINT(constr),          # 删除约束
    CHANGE_BOUND(var, lb, ub),        # 修改变量边界
    REWRITE(code_patch),              # 重写代码片段

    # Meta Actions
    SUBMIT(),               # 提交当前模型
    RESTART()               # 重新开始
}

Transition T(s'|s,a) = Deterministic (Solver执行)

Reward R(s,a,s') = R_outcome + R_process + R_faithfulness
```

---

## MDP在Bench中的体现

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Bench设计 ← MDP State要求                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MDP State需要什么?              Bench必须提供什么?                  │
│  ─────────────────────────────────────────────────────────────────  │
│  problem_nl                  →   每个问题必须有NL描述               │
│  model_code                  →   Saboteur生成的破坏代码              │
│  solver_status               →   必须是INFEASIBLE (四重验证)         │
│  iis_log                     →   IIS ground truth (关键!)            │
│  ground_truth_fix            →   用于计算Outcome reward              │
│                                                                      │
│  关键insight:                                                        │
│  - Bench的数据结构 = MDP State的具体化                              │
│  - 如果Bench缺少某个字段，MDP State就不完整                         │
│  - 例如: 没有IIS → 无法计算DA → Novelty 2无法体现                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Bench数据结构 (由MDP State定义)**:
```python
@dataclass
class BenchProblem:
    """每个字段对应MDP State的一个组件"""

    # MDP State组件
    problem_id: str
    problem_nl: str           # → State.problem_nl
    original_code: str        # 原始可行模型 (用于计算OP)
    sabotaged_code: str       # → State.model_code (初始)
    initial_status: str       # → State.solver_status (必须是INFEASIBLE)
    iis: List[str]            # → State.iis_log (ground truth)

    # MDP Reward计算需要
    error_type: str           # A/B/C/D (用于分层分析)
    target: str               # 被破坏的约束/变量
    ground_truth_fix: str     # 用于Outcome reward验证
    original_objective: float # 用于OP计算
    difficulty: str           # easy/medium/hard
```

---

## MDP在Eval中的体现

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Eval指标 ← MDP组件                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MDP组件                     Eval指标                               │
│  ─────────────────────────────────────────────────────────────────  │
│  Transition T              →   Recovery Rate (是否到达OPTIMAL)       │
│  State.step                →   Trajectory Efficiency (1/steps)       │
│  State.iis_log             →   Diagnosis Accuracy (pred ∩ actual)   │
│  Action sequence           →   RR@k (k步内是否恢复)                  │
│  Final model vs original   →   Optimality/Feasibility Preservation  │
│                                                                      │
│  关键insight:                                                        │
│  - 每个Eval指标都是MDP某个组件的函数                                │
│  - RR@k = f(Trajectory长度)                                          │
│  - DA = f(Agent诊断, State.iis_log)                                 │
│  - OP = f(Final objective, Original objective)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## MDP在Training中的体现

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Training ← MDP Reward                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MDP Reward组件              Training如何使用                        │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  R_outcome (sparse)      →   GRPO的主要信号                          │
│    +100 if OPTIMAL           - 轨迹结束时计算                        │
│    -50  if still INFEASIBLE  - Group内对比优化                       │
│                                                                      │
│  R_process (dense)       →   SFT数据筛选 + GRPO shaping              │
│    +10 if |IIS|减少          - 筛选"IIS逐步减少"的轨迹做SFT         │
│    -1  per step              - GRPO中鼓励高效轨迹                    │
│                                                                      │
│  R_faithfulness (novel)  →   防止"蒙对"                              │
│    -20 if 诊断与IIS无交集    - 即使成功也惩罚错误诊断                │
│                              - 这是OR-Debug-Bench的核心创新          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## MDP贯穿全流程总结

| 阶段 | MDP组件 | 具体对应 |
|------|---------|----------|
| **Bench** | State结构 | problem_nl, sabotaged_code, iis, solver_status |
| **Eval** | 指标计算 | RR←Transition, DA←iis_log, TE←step |
| **Training** | Reward设计 | Outcome←status, Process←step, Faithfulness←iis |

**一致性保证**:
- Bench的IIS字段 = Eval的DA分母 = Training的Faithfulness基准
- Bench的solver_status = Eval的RR判断 = Training的Outcome信号
- 三者使用相同的MDP定义，确保评估和训练的一致性

---

*相关文档*:
- [Novelty定位](02_NOVELTY.md)
- [连贯性设计](06_COHERENCE.md)
