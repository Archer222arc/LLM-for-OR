# Toy Model 项目规范

## 📋 概述

Toy Model是用于验证PPO路由策略的M/M/1队列仿真实验，**完全独立于Vidur主项目**，但可选择性复用核心PPO组件。

---

## 🎯 核心原则

### 1. 模块隔离原则
- **Toy Model代码**: 统一放在 `toymodel/` 根目录
- **Vidur代码**: 保持在 `vidur/` 和 `src/` 中
- **禁止交叉依赖**: Toy model不应被Vidur依赖

### 2. 复用策略
| 组件类型 | 复用方式 | 说明 |
|---------|---------|------|
| PPO算法核心 | ✅ 直接import `src.core.algorithms.PPOTrainer` | 算法通用 |
| ActorCritic网络 | ✅ 直接import `src.core.models.ActorCritic` | 网络通用 |
| 状态构建 | ❌ 独立实现 `toymodel.state_builder` | 状态空间不同 |
| 奖励计算 | ❌ 独立实现 `toymodel.reward` | 奖励函数不同 |
| 仿真环境 | ❌ 独立实现 `toymodel.environment` | M/M/1 vs 事件驱动 |

### 3. 接口设计原则
- **简洁优先**: Toy model状态维度 << Vidur状态维度
- **可测试性**: 每个组件独立可测
- **显式验证**: 禁止fallback，配置缺失直接报错

---

## 📁 目录结构规范

```
Vidur_toymodel/
├── toymodel/                    # Toy Model根目录 (独立模块)
│   ├── __init__.py
│   ├── environment.py           # M/M/1队列环境
│   ├── state_builder.py         # 状态构建
│   ├── reward.py                # 奖励计算
│   ├── schedulers/              # 路由策略
│   │   ├── __init__.py
│   │   ├── ppo_scheduler.py    # PPO路由
│   │   ├── oracle.py            # 最优策略
│   │   └── baselines.py         # Random/RR
│   ├── training/                # 训练流程
│   │   ├── __init__.py
│   │   └── trainer.py           # 主训练器
│   └── metrics/                 # 指标收集
│       ├── __init__.py
│       └── collector.py
│
├── configs/toymodel/            # Toy Model配置
│   ├── base.yaml
│   ├── balanced_load.yaml
│   └── high_load.yaml
│
├── demo/                        # 示例代码
│   └── demo_environment.py      # 环境使用演示
│
├── scripts/
│   ├── toymodel_train.sh        # 训练脚本
│   └── toymodel_eval.sh         # 评估脚本
│
├── tests/toymodel/              # Toy Model测试
│   ├── test_environment.py
│   └── test_integration.py
│
├── tmp/                         # 临时测试文件 (即测即删)
│   └── README.md                # 仅保留README
│
├── outputs/toymodel/            # Toy Model输出
│   ├── checkpoints/
│   ├── metrics/
│   └── tensorboard/
│
├── .claude/
│   └── TOYMODEL.md             # 本文档 (必读)
│
└── docs/
    ├── toymodel_ppo_routing_design.md  # 技术方案
    └── toymodel_implementation.md      # 实现文档
```

**关键点**:
- ✅ Toy model代码在 `toymodel/` 根目录，与 `vidur/` 和 `src/` 平级
- ✅ 配置/脚本/测试/输出都按 `toymodel/` 命名空间组织
- ✅ 临时测试文件统一放在 `tmp/`，用完即删
- ❌ 不在 `src/toymodel/` 下，避免与Vidur核心代码混淆

---

## 🔧 组件复用接口

### 复用PPO组件示例

```python
# toymodel/training/trainer.py

from src.core.models import ActorCritic
from src.core.algorithms import PPOTrainer

# 直接使用，仅调整维度和超参数
policy = ActorCritic(
    state_dim=6,              # Toy model简化状态
    action_dim=2,             # 2个replica
    hidden_size=64,           # 比Vidur更小
    layer_N=1,
    gru_layers=1,
    enable_decoupled=False,   # 使用标准架构
)

trainer = PPOTrainer(
    policy=policy,
    lr=3e-4,
    clip_ratio=0.2,
    minibatch_size=32,        # 比Vidur更小
)
```

### 独立实现状态构建

```python
# toymodel/state_builder.py

import numpy as np

class ToyStateBuilder:
    """Toy model状态构建器 (6维状态)."""

    def build_state(
        self,
        replica_queues: list[int],        # [q1, q2]
        replica_utilizations: list[float], # [u1, u2]
        current_request_type: int,         # 0 or 1
        time_since_last_arrival: float,
    ) -> np.ndarray:
        """构建状态向量 [q1, u1, q2, u2, type, time]."""
        return np.array([
            replica_queues[0],
            replica_utilizations[0],
            replica_queues[1],
            replica_utilizations[1],
            float(current_request_type),
            time_since_last_arrival,
        ], dtype=np.float32)
```

---

## 🚫 命名规范

### 禁用前缀/后缀
- ❌ `toymodel_enhanced_*`, `*_toymodel_v2`
- ❌ `simple_*`, `toy_*_simple` (避免贬低性命名)

### 推荐命名
- ✅ `toymodel/environment.py` (清晰模块名)
- ✅ `toymodel/schedulers/oracle.py` (功能导向)
- ✅ `configs/toymodel/balanced_load.yaml` (场景导向)

---

## ✅ 配置规范

### 配置文件结构

```yaml
# configs/toymodel/base.yaml

environment:
  num_replicas: 2
  service_rates:
    replica_0: {type_A: 10.0, type_B: 5.0}
    replica_1: {type_A: 5.0, type_B: 10.0}
  arrival_rates: {type_A: 6.0, type_B: 6.0}
  max_steps: 10000
  seed: 42

model:
  state_dim: 6        # 固定维度
  action_dim: 2       # 固定为2个replica
  hidden_size: 64
  layer_N: 1
  gru_layers: 1

ppo:
  learning_rate: 0.0003
  gamma: 0.99
  clip_ratio: 0.2
  entropy_coef: 0.01
  minibatch_size: 32
  rollout_length: 2048

reward:
  queue_weight: 1.0
  routing_bonus: 0.1

training:
  total_steps: 100000
  eval_interval: 1000
  checkpoint_interval: 5000
```

**配置验证**: 启动时必须显式验证所有必需字段，缺失直接报错。

---

## 🧪 测试规范

### 测试分类

**正式测试** (`tests/toymodel/`)：
- 单元测试和集成测试
- 使用pytest框架
- 提交到git版本控制
- 最低覆盖率要求: 核心组件 ≥ 80%, 适配器 ≥ 90%

**临时测试** (`tmp/`)：
- 快速验证和调试
- 用完立即删除
- 不提交到git (已在.gitignore)
- 生命周期 ≤ 1天

### 临时测试使用示例

```bash
# 1. 创建临时测试文件
cat > tmp/test_feature.py << 'EOF'
from toymodel.environment import QueueEnvironment
env = QueueEnvironment(...)
# Quick validation code
print("✓ Test passed")
EOF

# 2. 运行测试
python tmp/test_feature.py

# 3. 立即删除
rm tmp/test_feature.py
```

### 关键测试点
1. **环境测试**: reset/step正确性，队列稳定性
2. **状态测试**: 维度正确性，数值范围合理性
3. **奖励测试**: 路由正确性影响奖励
4. **集成测试**: PPO训练收敛，接近Oracle性能

```python
# tests/toymodel/test_integration.py (正式测试)

def test_ppo_vs_oracle():
    """验证PPO接近Oracle性能."""
    # 训练PPO
    ppo_metrics = train_and_eval_ppo(steps=50000)

    # 评估Oracle
    oracle_metrics = eval_oracle()

    # 验证性能 (10%容忍度)
    assert ppo_metrics["mean_latency"] <= oracle_metrics["mean_latency"] * 1.1
    assert ppo_metrics["routing_accuracy"] >= 0.9
```

---

## 📊 监控规范

### TensorBoard指标
- `train/reward`: 每步奖励
- `train/policy_loss`: Actor loss
- `train/value_loss`: Critic loss
- `train/entropy`: 策略熵
- `eval/routing_accuracy`: 路由准确率
- `eval/mean_latency`: 平均延迟
- `eval/p99_latency`: P99延迟

### CSV导出
- 每个checkpoint保存指标CSV到 `outputs/toymodel/metrics/`
- 包含: step, reward, latency, routing_accuracy, throughput

---

## 🔍 代码审查Checklist

提交前必检:
- [ ] 代码在 `toymodel/` 目录，未混入 `vidur/` 或 `src/`
- [ ] 无禁用命名前缀 (`enhanced_*`, `*_v2` 等)
- [ ] 配置参数经过显式验证，无fallback
- [ ] 单元测试覆盖率 ≥ 80%
- [ ] 集成测试通过 (PPO收敛)
- [ ] TensorBoard指标正常记录

---

## 📖 相关文档

- **技术方案**: `docs/toymodel_ppo_routing_design.md` - 详细设计文档
- **实现指南**: `docs/toymodel_implementation.md` - 开发步骤
- **主规范**: `.claude/CLAUDE.md` - 项目通用规范

---

**规范版本**: v1.0
**生效日期**: 2025-10-01
**必读**: 开发Toy Model相关功能前必须阅读本文档
