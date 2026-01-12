# 模块接口文档索引

本目录包含 OR-Debug-Bench 项目各核心模块的详细接口文档。

## 文档导航

### 核心模块

| 模块 | 文档 | 说明 |
|------|------|------|
| **Environments** | [environments.md](environments.md) | MDP环境实现，包含State/Action/Reward设计 |
| **Agents** | [agents.md](agents.md) | 代理实现，包含BaseAgent/LLM/Baseline agents |
| **Solvers** | [solvers.md](solvers.md) | 求解器接口，Gurobi封装与IIS提取 |
| **Data Generation** | [data_generation.md](data_generation.md) | 数据生成，Saboteur错误注入 |
| **Evaluation** | [evaluation.md](evaluation.md) | 评估指标，Metrics/Benchmark/Tracking |

### 工具模块

| 模块 | 文档 | 说明 |
|------|------|------|
| **Utils** | [utils.md](utils.md) | 工具函数，配置管理和日志 |

---

## 模块关系图

```
┌─────────────────┐
│  Environments   │◄────┐
│  (MDP 核心)      │      │
└────┬────────────┘      │
     │                   │
     │ 使用              │ 交互
     ▼                   │
┌──────────┐      ┌──────┴─────┐
│ Solvers  │      │   Agents   │
│(求解器)   │      │  (代理)     │
└──────────┘      └────────────┘
     ▲                   │
     │ 注入错误            │ 评估
     │                   ▼
┌──────────────┐   ┌────────────┐
│ Data Gen     │   │ Evaluation │
│(数据生成)     │   │  (评估)     │
└──────────────┘   └────────────┘
```

---

## 快速开始

### 1. MDP 环境使用

```python
from src.solvers import GurobiSolver
from src.environments import SolverDebugEnv

# 创建求解器和环境
solver = GurobiSolver.from_file("model.mps")
env = SolverDebugEnv(solver)

# 运行 episode
state, info = env.reset()
action = agent.act(state)
next_state, reward, done, _, _ = env.step(action)
```

详见：[environments.md](environments.md)

### 2. 代理实现

```python
from src.agents import HeuristicAgent, LLMAgent

# 启发式代理
agent = HeuristicAgent()

# LLM代理
llm_agent = LLMAgent(model="gpt-4", provider="openai")
```

详见：[agents.md](agents.md)

### 3. 数据生成

```python
from src.data_generation import SaboteurAgent

# 注入错误
saboteur = SaboteurAgent(solver, seed=42)
result = saboteur.inject_type_d()  # 添加冲突约束
```

详见：[data_generation.md](data_generation.md)

### 4. 评估运行

```python
from src.evaluation import BenchmarkRunner, BenchmarkProblem

# 运行基准测试
runner = BenchmarkRunner()
results = runner.run_benchmark(problems, agent)
print(runner.format_summary())
```

详见：[evaluation.md](evaluation.md)

---

## 文档规范

所有模块文档遵循统一结构：

- **概述**: 模块定位和核心功能
- **设计理念**: 架构决策和技术选型
- **核心接口**: 主要类和函数说明
- **数据结构**: 关键数据类型定义
- **使用模式**: 常见用法和最佳实践
- **扩展指南**: 如何添加新功能
- **测试策略**: 单元测试和集成测试
- **依赖关系**: 内部和外部依赖
- **参考文献**: 相关论文和文档

---

## 演进历史

模块的历史版本和重大变更记录在 [evolution/](evolution/) 目录中。

---

## 相关文档

- **研究方向文档**: [docs/directions/A_OR_Debug_Bench/](../directions/A_OR_Debug_Bench/)
- **实现进度日志**: [docs/progress/](../progress/)
- **项目规范**: [.claude/CLAUDE.md](../../.claude/CLAUDE.md)

---

*最后更新: 2026-01-11*
