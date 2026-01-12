# 核心Novelty与文献定位

*返回: [项目计划](../PROJECT_PLAN.md)*

---

## Novelty→实现→训练 全流程映射

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Novelty-Driven Implementation Flow                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Novelty 1: CorrectBench Gap        Novelty 2: Process vs Outcome           │
│       ↓                                   ↓                                  │
│  ┌──────────────┐                   ┌──────────────┐                        │
│  │ Saboteur设计 │                   │ MDP Environment │                     │
│  │ 4种错误类型  │                   │ 6项过程指标     │                     │
│  └──────┬───────┘                   └───────┬────────┘                      │
│         │                                   │                                │
│         └──────────────┬────────────────────┘                                │
│                        ↓                                                     │
│                 ┌──────────────┐                                            │
│                 │   Benchmark  │ ← 体现Novelty 1&2                          │
│                 │  数据集构建   │                                            │
│                 └──────┬───────┘                                            │
│                        ↓                                                     │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │              Baseline Evaluation (11+ Models)                 │           │
│  │  输出: 成功轨迹 + 失败案例分析 + Test-time Compute数据       │           │
│  └──────────────────────────┬───────────────────────────────────┘           │
│                             ↓                                                │
│         ┌─────────────────────────────────────────┐                         │
│         │         SFT 数据构建                     │                         │
│         │  从强模型(GPT-5.2, o4-mini)收集成功轨迹  │                         │
│         │  格式: Instruction + Thinking + Action   │                         │
│         └────────────────────┬────────────────────┘                         │
│                              ↓                                               │
│         ┌─────────────────────────────────────────┐                         │
│         │      Qwen3-8B SFT (LoRA)                │                         │
│         │  学习基本debugging pattern              │                         │
│         └────────────────────┬────────────────────┘                         │
│                              ↓                                               │
│  Novelty 3: RLVR Oracle     Novelty 4: Test-time Compute                    │
│       ↓                           ↓                                          │
│  ┌──────────────┐          ┌──────────────┐                                 │
│  │ 三层奖励设计 │          │ RR@k 实验   │                                 │
│  │ Solver反馈   │          │ 步数vs性能   │                                 │
│  └──────┬───────┘          └──────┬───────┘                                 │
│         │                         │                                          │
│         └──────────┬──────────────┘                                          │
│                    ↓                                                         │
│         ┌─────────────────────────────────────────┐                         │
│         │      Qwen3-8B GRPO Training             │                         │
│         │  Solver作为verifiable reward oracle     │                         │
│         └─────────────────────────────────────────┘                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心思想**: 每个实现决策都必须回溯到某个Novelty，否则就是无意义的工作量。

---

## Novelty 1: 填补Self-Correction Blind Spot (CorrectBench Gap)

**文献背景**: CorrectBench (arXiv 2025) 发现64.5%的blind spot rate，但**完全忽略OR领域**。

**我们的贡献**:
- OR领域首个self-correction评估基准
- Solver提供**deterministic, noise-free feedback** (vs 代码测试的binary pass/fail)
- IIS作为ground truth，将self-correction从"introspective guessing"转为"diagnostic reasoning"

**实现要点**:
```
State = (NL Problem, Code, Solver Status, IIS Log, History)
Ground Truth = Solver IIS output (不可challenge)
```

**论文Position**: "CorrectBench omits constrained optimization entirely—creating a blue ocean opportunity"

**实现指导 → Saboteur设计**:
```python
# Novelty 1要求: 错误必须产生可诊断的IIS
# 这决定了Saboteur的设计原则

class SaboteurDesignPrinciple:
    """
    每种错误类型必须满足:
    1. 注入后状态 = INFEASIBLE (可验证)
    2. IIS非空且包含target (可诊断)
    3. 存在唯一正确修复 (ground truth)
    """

    # Type A: 方向翻转 → 测试"理解约束语义"能力
    # Novelty体现: 需要理解NL描述来判断哪个方向是正确的

    # Type D: 直接矛盾 → 测试"识别逻辑冲突"能力
    # Novelty体现: IIS直接给出答案，测试基础诊断能力

    # Type B/C: 复杂错误 → 测试"深度推理"能力
    # Novelty体现: IIS只给线索，需要理解变量语义
```

---

## Novelty 2: Process vs Outcome评估 (MDP Policy)

**文献背景**:
- 现有Benchmark: NL4Opt, OptiBench, IndustryOR都是**静态Pass@1评估**
- StepORLM: 提出GenPRM做过程监督，但只用于formulation，**未涉及debugging**

**我们的贡献**:
- 从评估`P(correct code | text)`转向评估`π(aₜ | sₜ)`
- 新指标体系: Recovery Rate@k, Trajectory Efficiency, Diagnosis Accuracy
- **首个OR领域的Gym-like MDP环境**

**指标创新**:

| 指标 | 公式 | 衡量什么 | 对标文献 |
|------|------|----------|----------|
| RR@k | 在k步内恢复的比例 | Debugging能力 | SWE-bench的resolve rate |
| DA | \|pred∩actual_IIS\|/\|actual_IIS\| | 推理质量 | StepORLM的step correctness |
| TE | 1/steps | 思考效率 | Test-time compute效率 |
| OP | 1-\|ΔZ\|/\|Z_orig\| | 修复质量 | 惩罚trivial fix |
| FP | 保留约束比例 | 保守性 | 避免"delete all" |

**实现指导 → 指标如何反映Novelty**:
```python
# 每个指标对应一个研究问题

class MetricNoveltyMapping:
    """
    RR@k: 回答"Test-time compute是否有效?"
        - 对比RR@5 vs RR@20，quantify更多step的边际收益
        - 对比Reasoning model vs Standard model的RR曲线

    DA: 回答"模型是真的理解还是蒙对?"
        - 高RR + 低DA = 蒙对 (Faithfulness问题)
        - 高RR + 高DA = 真正理解

    OP/FP: 回答"修复质量如何?"
        - 防止trivial fix (删除所有约束)
        - 这是OR领域特有的评估维度
    """

# 论文Table设计
TABLE_1_DESIGN = {
    "行": ["GPT-5.2", "o4-mini", "DeepSeek-R1", "Qwen3-8B", "Qwen3-8B-SFT", "Qwen3-8B-GRPO"],
    "列": ["RR@5", "RR@10", "RR@20", "DA", "OP", "TE"],
    "故事": "GRPO提升RR的同时保持高DA，证明不是蒙对"
}
```

---

## Novelty 3: Solver as RLVR Oracle

**文献背景**:
- DeepSeek-R1 (Nature 2025): GRPO纯RL诱导reasoning，**需要verifiable rewards**
- OR-R1: TGRPO在OR任务上+6%，验证OR是RLVR的理想领域
- Tülu 3: RLVR在math上有效

**我们的贡献**:
- Solver作为**perfect oracle**: 无噪声、确定性、可解释
- 首次将GRPO应用于OR debugging任务
- 设计**Outcome + Process + Faithfulness**三层奖励

**奖励函数设计** (核心Novelty):
```python
R(trajectory) = R_outcome + R_process + R_faithfulness

R_outcome:
  - +100 if final status == OPTIMAL
  - -50  if final status == INFEASIBLE

R_process (dense, per-step):
  - +10 if |IIS_t+1| < |IIS_t|  # IIS缩小
  - -1  per step               # 效率惩罚

R_faithfulness (novel):
  - -20 if agent_diagnosis ∩ actual_IIS == ∅  # 诊断错误
```

**为什么Faithfulness重要**: 防止模型"蒙对"——即使最终修复成功，如果诊断完全错误也应惩罚

**实现指导 → SFT/GRPO训练流程**:
```python
# ============================================
# Stage 1: SFT数据收集 (从Baseline评估中获取)
# ============================================

def collect_sft_data_from_evaluation():
    """
    SFT数据来源: Baseline评估中强模型的成功轨迹

    关键: 不是随便收集，而是有策略地筛选
    """
    teacher_models = ["gpt-5.2-chat", "o4-mini", "DeepSeek-R1"]
    sft_data = []

    for problem in benchmark_problems:
        for model in teacher_models:
            trajectory = run_episode(model, problem)

            # 筛选条件 (体现Novelty):
            if (trajectory.success and           # 必须成功
                trajectory.steps <= 5 and        # 效率高
                trajectory.diagnosis_accuracy >= 0.5):  # 诊断准确
                sft_data.append(format_trajectory(trajectory))

    return sft_data  # 目标: 3000-5000条高质量轨迹

def format_trajectory(traj) -> dict:
    """
    SFT格式: 强调reasoning过程

    这是体现Novelty的关键:
    - 不只是(input, output)对
    - 而是(input, thinking, action)三元组
    """
    return {
        "instruction": "Debug the infeasible optimization model.",
        "input": f"""
Problem: {traj.problem_nl}
Code: {traj.code}
Status: INFEASIBLE
IIS: {traj.iis}
""",
        "output": f"""
<think>
Step 1: Analyze IIS - {traj.iis_analysis}
Step 2: Identify root cause - {traj.root_cause}
Step 3: Consider fix options - {traj.fix_options}
Step 4: Choose best fix - {traj.chosen_fix_reason}
</think>

Diagnosis: {traj.diagnosis}
Action: {traj.action}
"""
    }

# ============================================
# Stage 2: GRPO奖励设计 (体现Novelty 3)
# ============================================

class GRPORewardCalculator:
    """
    三层奖励 = Outcome + Process + Faithfulness

    关键创新: Faithfulness penalty
    """

    def compute_reward(self, trajectory):
        r_outcome = self.outcome_reward(trajectory)
        r_process = self.process_reward(trajectory)
        r_faith = self.faithfulness_penalty(trajectory)

        return r_outcome + r_process + r_faith

    def faithfulness_penalty(self, trajectory):
        """
        这是本文的核心创新之一

        如果agent的诊断与实际IIS完全无交集:
        → 即使最终修复成功，也要惩罚
        → 这鼓励模型"真正理解"而非"蒙对"
        """
        agent_diagnosis = extract_diagnosis(trajectory.output)
        actual_iis = trajectory.ground_truth_iis

        if agent_diagnosis.isdisjoint(actual_iis):
            return -20  # 严厉惩罚
        return 0
```

---

## Novelty 4: Test-Time Compute Evaluation

**文献背景**:
- OpenAI o3 (2026): Test-time compute scaling，71.7% on SWE-bench
- Scaling Test-Time Compute (ICLR 2025): 小模型+更多推理时间 > 大模型

**我们的贡献**:
- RR@k指标直接衡量test-time compute效率
- 实验对比: Reasoning模型(o1, DeepSeek-R1) vs 普通模型(GPT-4)
- 分析"thinking quality"而非只看最终结果

**实验设计**:
```
对比实验:
1. GPT-4.1 (fast thinking) vs o4-mini (slow thinking)
2. 相同模型，不同max_steps限制 (5, 10, 20)
3. 分析: 更多step是否带来更好RR，还是只是浪费?
```

**实现指导 → 实验设计**:
```python
# ============================================
# Test-Time Compute实验矩阵
# ============================================

EXPERIMENT_MATRIX = {
    # 维度1: 模型类型
    "models": {
        "reasoning": ["o1", "o4-mini", "DeepSeek-R1"],
        "standard": ["gpt-4.1", "gpt-5-mini", "Qwen3-8B"],
        "ours": ["Qwen3-8B-SFT", "Qwen3-8B-GRPO"]
    },

    # 维度2: Step budget
    "max_steps": [5, 10, 15, 20],

    # 维度3: 问题难度
    "difficulty": ["easy", "medium", "hard"]
}

def run_test_time_compute_experiment():
    """
    核心问题: Test-time compute在OR debugging中是否有效?

    预期发现:
    1. Reasoning模型在Hard问题上收益大
    2. Standard模型在Easy问题上已经饱和
    3. GRPO训练让Qwen3-8B接近Reasoning模型水平
    """
    results = {}

    for model in all_models:
        for max_steps in [5, 10, 15, 20]:
            rr = evaluate_model(model, max_steps)
            results[(model, max_steps)] = rr

    # 绘制图表
    plot_rr_vs_steps(results)  # Figure 1 in paper

    return results

# 论文Figure设计
FIGURE_1 = {
    "title": "Test-Time Compute Scaling in OR Debugging",
    "x_axis": "Max Steps Allowed",
    "y_axis": "Recovery Rate (%)",
    "lines": ["o4-mini", "gpt-4.1", "Qwen3-8B-GRPO"],
    "story": "GRPO使小模型具备类似reasoning模型的test-time scaling特性"
}
```

---

*相关文档*:
- [MDP设计](01_MDP_DESIGN.md)
- [评估框架](04_EVAL.md)
- [训练流程](05_TRAINING.md)
