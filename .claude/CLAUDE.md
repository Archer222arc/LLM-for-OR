# Claude Code 项目规范 - LLM-for-OR

## 🎯 项目定位

**项目名称**: LLM-for-OR (NeurIPS 2026 Agentic OR/OM Research)

**研究领域**:
- Operations Research (OR) - 数学优化、求解器调试
- Operations Management (OM) - 供应链、库存管理、服务运营
- LLM Agents - 大语言模型代理决策

**核心目标**:
- 构建动态自我纠错的OR代理评估基准（OR-Debug-Bench）
- 从静态翻译（NL→Code）转向过程评估（MDP Policy）

**项目计划文档**:
| 文档 | 内容 |
|------|------|
| [`docs/plan/PROJECT_PLAN.md`](../docs/plan/PROJECT_PLAN.md) | 总计划索引（精简版） |
| [`docs/plan/modules/01_MDP_DESIGN.md`](../docs/plan/modules/01_MDP_DESIGN.md) | MDP设计与各部分定位 |
| [`docs/plan/modules/02_NOVELTY.md`](../docs/plan/modules/02_NOVELTY.md) | 核心Novelty与文献定位 |
| [`docs/plan/modules/03_BENCH.md`](../docs/plan/modules/03_BENCH.md) | Phase 1: 数据构建 |
| [`docs/plan/modules/04_EVAL.md`](../docs/plan/modules/04_EVAL.md) | Phase 2: 评估框架 |
| [`docs/plan/modules/05_TRAINING.md`](../docs/plan/modules/05_TRAINING.md) | Phase 3: RLVR训练 |
| [`docs/plan/modules/06_COHERENCE.md`](../docs/plan/modules/06_COHERENCE.md) | Bench→Eval→Training连贯性 |

**研究方向文档** (高优先级阅读):
| 文档 | 内容 |
|------|------|
| [`docs/research/00_PROJECT_OVERVIEW.md`](../docs/research/00_PROJECT_OVERVIEW.md) | 项目总览、方向对比 |
| [`docs/research/01_RESEARCH_CONTEXT.md`](../docs/research/01_RESEARCH_CONTEXT.md) | 研究背景、RLVR/PRM |
| [`docs/research/02_OPERATIONS_MANAGEMENT.md`](../docs/research/02_OPERATIONS_MANAGEMENT.md) | OM扩展方向 |
| [`docs/research/03_LITERATURE_REVIEW.md`](../docs/research/03_LITERATURE_REVIEW.md) | 70+文献综述 |
| [`docs/research/04_IMPLEMENTATION_ROADMAP.md`](../docs/research/04_IMPLEMENTATION_ROADMAP.md) | 实施路线图 |

**研究方向（8个）**:
| Tier | 方向 | 名称 | 优先级 |
|------|------|------|--------|
| 1 | A | OR-Debug-Bench | ⭐⭐⭐ 主要 |
| 1 | B | OR-Bias-Bench | ⭐⭐ 次要 |
| 1 | C | OR-Compliance-Bench | ⭐⭐ 次要 |
| 2 | D-F | Formulation/Transfer/Disruption | ⭐ 扩展 |
| 3 | G-H | Safety-RL/Multi-Agent | ⭐ 远期 |

---

## 📁 标准目录结构

```
LLM-for-OR/
├── docs/                          # 📚 研究文档
│   ├── plan/                      # 项目计划
│   │   ├── PROJECT_PLAN.md        # 总计划索引（精简版）
│   │   └── modules/               # 详细模块文档
│   │       ├── 01_MDP_DESIGN.md   # MDP设计
│   │       ├── 02_NOVELTY.md      # Novelty定位
│   │       ├── 03_BENCH.md        # 数据构建
│   │       ├── 04_EVAL.md         # 评估框架
│   │       ├── 05_TRAINING.md     # 训练流程
│   │       └── 06_COHERENCE.md    # 连贯性设计
│   ├── research/                  # 研究方向文档（高优先级）
│   │   ├── 00_PROJECT_OVERVIEW.md # 项目总览
│   │   ├── 01_RESEARCH_CONTEXT.md # 研究背景
│   │   ├── 02_OPERATIONS_MANAGEMENT.md # OM扩展
│   │   ├── 03_LITERATURE_REVIEW.md # 文献综述
│   │   └── 04_IMPLEMENTATION_ROADMAP.md # 实施路线图
│   ├── directions/                # 方向详细规范
│   │   ├── A_OR_Debug_Bench/      # 方向A（核心）
│   │   └── ...
│   ├── progress/                  # 进度日志
│   └── modules/                   # 模块接口文档
│
├── src/                           # 🔧 核心代码（按技术组件）
│   ├── environments/              # MDP环境
│   ├── agents/                    # 代理实现
│   ├── solvers/                   # 求解器接口
│   ├── data_generation/           # 数据生成
│   ├── evaluation/                # 评估指标
│   └── utils/                     # 工具函数
│
├── configs/                       # ⚙️ 配置文件
│   ├── benchmarks/                # 基准配置
│   ├── experiments/               # 实验配置
│   └── models/                    # 模型配置
│
├── data/                          # 📊 数据
│   ├── raw/                       # 原始数据（MIPLIB, OptMATH）
│   ├── processed/                 # 处理后数据
│   ├── synthetic/                 # 合成数据（Saboteur生成）
│   └── benchmarks/                # 基准数据集
│
├── experiments/                   # 🧪 实验配置
├── outputs/                       # 📈 输出（git忽略）
├── scripts/                       # 🚀 脚本（模块化组织）
│   ├── data_generation/           # 数据生成脚本
│   │   └── generate_dataset.py    # Benchmark数据集生成
│   ├── evaluation/                # 评估脚本
│   │   ├── evaluate_llm.py        # LLM评估主脚本（支持SQLite）
│   │   ├── analyze_results.py     # 结果分析与可视化
│   │   └── validate_robust_methods.py  # Robust方法验证
│   ├── training/                  # 训练脚本
│   │   ├── collect_sft_data.py    # SFT数据收集
│   │   └── run_llm_experiment.py  # LLM实验运行
│   ├── experiments/               # 实验运行脚本（bash）
│   │   ├── run_llm_eval.sh        # LLM评估主脚本
│   │   └── monitor_eval.sh        # 进度监控脚本
│   ├── deployment/                # 部署脚本
│   │   ├── azure/                 # Azure部署（deploy_models.sh等）
│   │   └── foundry/               # Foundry部署（guide_deployment.py等）
│   ├── utils/                     # 工具脚本
│   │   └── verify_installation.py # 环境安装验证
│   └── visualization/             # 可视化脚本（预留）
├── notebooks/                     # 📓 Jupyter分析
├── tests/                         # ✅ 测试
├── demo/                          # 🎯 示例
├── logs/                          # 📝 日志（git忽略）
└── tmp/                           # 🗑️ 临时文件（git忽略）
```

---

## 🚨 核心编程规范

### 八荣八耻编程原则

1. **以暗猜接口为耻，以认真查阅为荣** - 禁止臆测API行为，必须查阅文档确认
2. **以模糊执行为耻，以寻求确认为荣** - 不确定的实现必须先向用户确认
3. **以默认忽略为耻，以主动报告为荣** - 遇到异常、错误必须主动报告
4. **以隐式假设为耻，以显式验证为荣** - 所有假设必须通过代码验证
5. **以随意修改为耻，以谨慎调试为荣** - 修改前必须理解原理
6. **以表面应付为耻，以深入理解为荣** - 解决问题必须找到根本原因
7. **以复制粘贴为耻，以原创思考为荣** - 理解每行代码含义
8. **以孤立开发为耻，以协同沟通为荣** - 主动汇报进度和问题

### 🔥 文件命名规范

**禁用前缀后缀列表**：
- ❌ `enhanced_*` / `*_enhanced`
- ❌ `integrated_*` / `*_integrated`
- ❌ `cleaned_*` / `*_clean`
- ❌ `improved_*` / `*_improved`
- ❌ `*_v2` / `*_new` / `*_old` / `*_temp`

**正确命名原则**：
- ✅ **功能导向**: `solver_gym.py`, `saboteur_agent.py`
- ✅ **模块化**: `environments/`, `agents/`, `solvers/`
- ✅ **简洁明确**: 使用下划线分隔，全小写

### 🛡️ 错误处理规范

```python
# ❌ 禁止fallback模式
try:
    result = complex_operation()
except Exception:
    result = fallback_operation()  # 禁止！

# ✅ 让错误自然抛出
result = complex_operation()  # 便于从本质上解决问题
```

**核心要求**：
- 🔥 **禁止静默捕获异常** - 让错误traceback显示
- 🔥 **禁止fallback方案** - 缺少属性直接报错

### 📊 可视化规范

**语言要求**：
- 🔥 **所有图表文字使用英文** - 标题、轴标签、图例、注释全部使用English
- 面向国际发表，确保学术论文可直接使用

**样式要求**：
- **DPI**: 300（出版质量）
- **字体**: Sans-serif (DejaVu Sans 或类似)
- **配色**: Seaborn "colorblind" 色盲友好调色板
- **图片尺寸**: 默认 10x6 或 12x7 英寸
- **网格**: 浅灰色虚线

**脚本位置**: `scripts/visualization/`

**输出位置**: `outputs/experiments/{experiment}/figures/`

**命名规范**:
- ✅ `fig1_performance_overview.png`
- ✅ `fig2_token_efficiency.png`
- ❌ `图1_性能对比.png` (禁止中文文件名)

---

## 🎮 MDP形式化规范

### State定义标准
每个Benchmark的State必须包含：
- **Problem**: 原始问题描述（NL + Code）
- **Status**: 求解器状态（Optimal/Infeasible/Unbounded/Error）
- **Feedback**: 求解器反馈（IIS/Slack/Gap/ErrorLog）
- **History**: 修改历史轨迹

### Action定义标准
分层Action Space：
- **Diagnosis**: `Get_IIS()`, `Check_Slack()`, `Explain_Conflict()`
- **Repair**: `Relax_Constraint()`, `Drop_Constraint()`, `Rewrite()`
- **Meta**: `Restart`, `Submit`, `Query_User`

### Reward定义标准
组合奖励函数：
- Outcome Reward: 可行性恢复 (+100/-50)
- Process Reward: IIS缩小 (+10), 步数惩罚 (-1)
- Faithfulness Penalty: 诊断与日志矛盾 (-20)

---

## 🔧 求解器代码规范

### Gurobi代码风格
```python
import gurobipy as gp

# Model命名
m = gp.Model("problem_name")

# 变量使用元组索引
x = m.addVars(nodes, nodes, vtype=gp.GRB.BINARY, name="x")

# 约束必须命名
m.addConstr(gp.quicksum(x[i,j] for j in nodes) == 1, name=f"assign_{i}")
```

### IIS提取规范
```python
# 正确的IIS提取方式
m.computeIIS()
iis_constraints = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
iis_bounds = [v.VarName for v in m.getVars() if v.IISLB or v.IISUB]
```

### Pyomo代码风格
```python
from pyomo.environ import *

# 使用ConcreteModel
model = ConcreteModel()

# Solver统一使用SolverFactory
solver = SolverFactory('gurobi')
results = solver.solve(model)
```

---

## 🧪 实验可复现规范

### 实验运行脚本

**核心原则**：使用结构化bash脚本运行实验，便于复现和手动执行

**脚本位置**: `scripts/experiments/`

| 脚本 | 用途 |
|------|------|
| `run_llm_eval.sh` | LLM评估主脚本 |
| `monitor_eval.sh` | 实时监控进度 |

**使用示例**:
```bash
# 单模型评估
./scripts/experiments/run_llm_eval.sh --model gpt-5.2-chat --samples 200

# 全模型并行评估
./scripts/experiments/run_llm_eval.sh --all --samples 200 --parallel

# 断点续传
./scripts/experiments/run_llm_eval.sh --model o4-mini --samples 500 --resume

# 监控进度
./scripts/experiments/monitor_eval.sh --watch
```

### 实验配置要求
每个实验自动保存：
- `config.yaml`: 完整配置参数（自动生成）
- `git_hash.txt`: 代码版本（自动生成）
- `results.db`: SQLite结果数据库
- `results.json`: 导出的JSON结果
- `logs/`: 每个模型的运行日志

### 结果记录要求
- 主指标: Recovery Rate (RR@k), Diagnosis Accuracy (DA), Optimality Preservation (OP)
- 过程数据: 每步State/Action/Reward（存储于SQLite）
- 增量保存: 每个问题完成后立即写入数据库

### Outputs目录结构

**核心原则**：按日期和实验名组织，避免文件堆积

```
outputs/
├── experiments/                    # 结构化实验输出
│   ├── 2026-01-12/                # 按日期分组
│   │   ├── all_models_200samples/ # 实验名
│   │   │   ├── config.yaml        # 实验配置
│   │   │   ├── git_hash.txt       # 代码版本
│   │   │   ├── results.db         # SQLite数据库
│   │   │   ├── results.json       # 导出的JSON
│   │   │   └── logs/              # 运行日志
│   │   │       ├── gpt-5.2-chat.log
│   │   │       ├── o4-mini.log
│   │   │       └── ...
│   │   └── o4mini_500samples/
│   │       └── ...
│   └── 2026-01-13/
│       └── ...
├── results.db                      # 默认数据库（快速测试用）
└── analysis/                       # 分析结果
    ├── figures/
    └── tables/
```

**命名规范**:
- 实验目录: `{模型或all}_{样本数}samples` 或自定义 `--exp-name`
- 日志文件: `{model_name}.log`
- 避免时间戳后缀，使用日期目录分组

### 禁止的输出文件命名
- ❌ `llm_gpt4_v2.json` - 使用日期目录代替版本号
- ❌ `results_20260112_153045.json` - 避免时间戳后缀
- ❌ `test_output.json` - 临时文件放入 `tmp/`

---

## 📚 研究方向文档规范

### 文档位置
`docs/directions/[DIRECTION_ID]/`

### 必需文件（方向A示例）
```
docs/directions/A_OR_Debug_Bench/
├── A1_Overview.md          # 方向总览
├── A2_MDP_Spec.md          # MDP形式化定义
├── A3_Data_Generation.md   # 数据生成策略
└── A4_Evaluation_Metrics.md # 评估指标定义
```

### 模块文档头部模板
```python
"""
[模块一句话描述]

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/

Key Components:
    - Component1: Brief description
    - Component2: Brief description

Example:
    >>> from module import Class
    >>> obj = Class()
    >>> result = obj.method()
"""
```

---

## 🗑️ 临时文件管理

**核心原则**：临时测试文件必须在 `tmp/` 目录，使用后立即删除

**强制要求**：
- 🚫 **禁止提交到git**: tmp/目录必须在.gitignore中
- 🚫 **禁止长期保留**: 文件生命周期 ≤ 1天
- 🚫 **禁止依赖关系**: 正式代码不得import tmp/中的文件

---

## 📝 TODO管理

### 项目级TODO
位置：`docs/PROJECT_TODO.md`

### 优先级定义
- **P0**: 阻塞开发/严重bug，立即处理
- **P1**: 新功能/重构，本周完成
- **P2**: 代码清理/文档补充，本月完成

### 使用规范
- 会话开始前查看 P0/P1 任务
- 会话中使用 TodoWrite 工具管理当前任务
- 会话结束后更新 PROJECT_TODO.md

---

## 🔗 关键引用

### 核心参考 (Direction A)
| 文献 | 出处 | 关联 |
|------|------|------|
| CorrectBench | arXiv 2025 | 64.5% blind spot，**完全忽略OR领域** |
| StepORLM | OpenReview 2025 | OR过程监督蓝图，dual-feedback机制 |
| RLVR/Tülu 3 | Allen AI 2024 | Solver作为可验证奖励oracle |
| DeepSeek-R1 | Nature 2025 | GRPO算法，纯RL诱导推理 |
| MOID | arXiv 2025 | 多目标infeasibility诊断 |

### 基准与框架
| 文献 | 出处 | 关联 |
|------|------|------|
| PILOT-Bench | ICLR 2026 | 工具驱动workflow框架 |
| NL4Opt | NeurIPS 2022 | NL→LP翻译竞赛 |
| OptiBench | arXiv 2024 | 建模准确率基准 |
| IndustryOR | arXiv 2024 | 工业复杂度案例 |

### 方法论
| 文献 | 出处 | 关联 |
|------|------|------|
| BiPRM | arXiv 2025 | 双向过程奖励模型 |
| PAVs | ICLR 2025 | Process Advantage Verifiers |
| Math-Shepherd | ACL 2024 | MC估计的自动过程监督 |

---

## 📅 项目进展日志

**位置**: `docs/progress/`

**命名规范**: `YYYY-MM-DD_milestone_name.md`

**内容规范**:
- 状态（完成/进行中）
- 完成内容列表
- 遇到问题及解决方案
- 产出统计（文件/行数）
- 下一步待办

---

## 🤖 训练基础设施 (GPU VM专属)

> ⚠️ **注意**：本节内容仅适用于配备A100 GPU的Azure VM。

### 模型训练流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Qwen3-8B   │ --> │  SFT训练    │ --> │  GRPO训练   │
│  (基础模型)  │     │  (696样本)  │     │  (44提示)   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
 /data/Qwen3-8B/   /data/qwen3_or_      /data/qwen3_or_
                   debug_merged/        debug_grpo_merged/
```

### 训练产出模型

| 模型 | 路径 | 大小 | 说明 |
|------|------|------|------|
| 基础模型 | `/data/Qwen3-8B/` | 16GB | Qwen3-8B原始权重 |
| SFT模型 | `/data/qwen3_or_debug_merged/` | 16GB | SFT微调后合并模型 |
| GRPO模型 | `/data/qwen3_or_debug_grpo_merged/` | 16GB | GRPO训练后合并模型 |
| SFT适配器 | `/data/sft_output/` | 2.4GB | LoRA适配器 |
| GRPO适配器 | `/data/grpo_output/final/` | 174MB | LoRA适配器 |

### 训练配置

**SFT训练** (2026-01-13完成):
- 方法: LoRA (rank=32, alpha=64)
- 数据: 696样本
- 最终Loss: 0.0676

**GRPO训练** (2026-01-13完成):
- 方法: LoRA (rank=16, alpha=32)
- 框架: TRL 0.24.0 GRPOTrainer
- 数据: 44个唯一提示
- 观察: 奖励方差为0 (SFT已达最优)

### 评估结果

| 指标 | SFT模型 | GRPO模型 |
|------|---------|----------|
| RR@5 | 83% | 83% |
| RR@10 | 99% | 99% |
| RR | 100% | 100% |
| DA | 80% | 80% |

> **结论**: SFT和GRPO性能一致，因GRPO训练时奖励方差为零，无梯度更新。

### 推理部署

**SGLang服务器** (高性能推理):
```bash
# 激活环境
source /data/envs/sglang/bin/activate

# 启动服务 (双A100)
python -m sglang.launch_server \
  --model-path /data/qwen3_or_debug_merged \
  --tensor-parallel-size 2 \
  --port 30000
```

详细配置见: `~/.claude/CLAUDE.md` → "GPU VM专属配置 - SGLang部署"

### 训练脚本位置

| 脚本 | 用途 |
|------|------|
| `scripts/training/train_grpo_trl.py` | GRPO训练 (TRL框架) |
| `scripts/training/prepare_grpo_data.py` | GRPO数据准备 |
| `scripts/training/merge_grpo_adapter.py` | LoRA适配器合并 |
| `scripts/training/collect_sft_data.py` | SFT数据收集 |

### 评估脚本位置

| 脚本 | 用途 |
|------|------|
| `scripts/evaluation/evaluate_local_model.py` | 本地模型评估 |
| `scripts/evaluation/compare_models.py` | SFT vs GRPO对比 |

---

*最后更新: 2026-01-13*
*项目版本: v0.6.0*
