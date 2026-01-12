# Phase 3: RLVR训练

*返回: [项目计划](../PROJECT_PLAN.md)*

---

## 3.1 SFT阶段 (Week 5-6)

**目标**: 让Qwen3-8B学会基本的debugging pattern

### 数据收集策略
```python
def collect_sft_data():
    """从强模型收集成功轨迹"""
    teacher_models = ["gpt-5.2-chat", "o4-mini"]
    trajectories = []

    for model in teacher_models:
        for problem in train_problems:
            traj = run_episode(model, problem)
            if traj.success and traj.steps <= 5:  # 只要高质量轨迹
                trajectories.append(format_as_sft(traj))

    return trajectories  # 目标: 3000-5000条
```

### SFT数据格式 (强调reasoning)
```json
{
  "instruction": "Debug the infeasible optimization model.",
  "input": "Model: [code]\nStatus: INFEASIBLE\nIIS: [c1, c2, c3]",
  "output": "<think>\n1. IIS contains c1, c2, c3\n2. c1 requires x >= 10\n3. c2 requires x <= 5\n4. These directly conflict\n5. c2 is a capacity constraint (more fundamental)\n</think>\n\nDiagnosis: c1 conflicts with c2\nAction: DROP_CONSTRAINT c1"
}
```

### SFT训练配置
```yaml
model:
  base: Qwen/Qwen3-8B-Instruct

lora:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  batch_size: 4
  gradient_accumulation: 8  # 有效batch=32
  learning_rate: 2e-4
  epochs: 3
  warmup_ratio: 0.1

# 显存估算: ~46GB (适合单卡A800)
```

---

## 3.2 GRPO阶段 (Week 7-8)

**核心创新**: Solver-backed verifiable rewards

### 实现架构
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Policy     │────>│  Rollout    │────>│  Solver     │
│  (Qwen3)    │     │  Sampler    │     │  (Gurobi)   │
└─────────────┘     └─────────────┘     └─────────────┘
       ^                   │                   │
       │                   v                   v
       │            ┌─────────────┐     ┌─────────────┐
       └────────────│  GRPO       │<────│  Reward     │
                    │  Update     │     │  Calculator │
                    └─────────────┘     └─────────────┘
```

### GRPO配置 (针对2×A800优化)
```yaml
grpo:
  group_size: 4              # 每问题4条轨迹
  max_steps_per_episode: 10  # 限制rollout长度
  kl_coef: 0.01              # KL散度系数

training:
  per_device_batch_size: 1   # 小batch避免OOM
  gradient_accumulation: 32  # 有效batch=32
  learning_rate: 1e-5
  total_steps: 5000

  # DeepSpeed ZeRO-2 (跨2卡)
  deepspeed:
    zero_stage: 2
    offload_optimizer: cpu

# 显存估算: ~52GB (需要2卡)
```

---

## 3.3 Process Reward Model (可选扩展)

**文献依据**: StepORLM, BiPRM, Math-Shepherd

### 实现思路
```python
class ProcessRewardModel:
    """
    预测每步action的质量
    输入: (state, action, next_state)
    输出: step_score ∈ [0, 1]
    """
    def __init__(self):
        self.model = load_classifier("Qwen3-1.5B")  # 小模型

    def score_step(self, state, action, next_state):
        # 特征: IIS变化、约束数变化、solver status
        features = extract_features(state, action, next_state)
        return self.model(features)
```

### 训练数据: 用solver结果自动标注
```
if |IIS_t+1| < |IIS_t|: label = 1  # 好的step
elif status changed to OPTIMAL: label = 1
else: label = 0
```

---

*相关文档*:
- [Novelty定位](02_NOVELTY.md) - RLVR Oracle (Novelty 3)
- [连贯性设计](06_COHERENCE.md) - Bench如何指导Training
