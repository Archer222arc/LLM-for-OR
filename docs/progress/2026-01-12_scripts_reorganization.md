# 2026-01-12: Scripts目录模块化重组 & 评估框架增强

## 状态: 完成

---

## 一、Scripts目录重组

### 1.1 重组前状态
- 14个脚本文件平铺在 `scripts/` 根目录
- 空子目录存在但未使用: `data_generation/`, `evaluation/`, `training/`, `visualization/`
- 难以快速定位和管理相关脚本

### 1.2 重组后结构

```
scripts/
├── data_generation/
│   └── generate_dataset.py          # 643行, Benchmark数据集生成
├── evaluation/
│   ├── evaluate_llm.py              # 698行, LLM评估主脚本
│   ├── analyze_results.py           # 310行, 结果分析
│   └── validate_robust_methods.py   # 391行, 方法验证
├── training/
│   ├── collect_sft_data.py          # 399行, SFT数据收集
│   └── run_llm_experiment.py        # 673行, 实验运行
├── deployment/
│   ├── azure/
│   │   ├── deploy_models.sh         # Azure模型部署
│   │   ├── setup_env.sh             # 环境设置
│   │   └── test_connection.py       # 115行, 连接测试
│   └── foundry/
│       ├── deploy_infrastructure.sh # 基础设施部署
│       ├── guide_deployment.py      # 350行, 部署指南
│       ├── update_config.py         # 290行, 配置更新
│       └── verify_deployment.py     # 166行, 部署验证
├── utils/
│   └── verify_installation.py       # 507行, 安装验证
└── visualization/                   # (预留)
```

### 1.3 命名简化
| 原文件名 | 新路径 |
|---------|--------|
| `deploy_azure_models.sh` | `deployment/azure/deploy_models.sh` |
| `setup_azure_env.sh` | `deployment/azure/setup_env.sh` |
| `test_azure_connection.py` | `deployment/azure/test_connection.py` |
| `deploy_foundry_infrastructure.sh` | `deployment/foundry/deploy_infrastructure.sh` |
| `guide_foundry_model_deployment.py` | `deployment/foundry/guide_deployment.py` |
| `update_foundry_config.py` | `deployment/foundry/update_config.py` |
| `verify_foundry_deployment.py` | `deployment/foundry/verify_deployment.py` |

---

## 二、评估框架增强

### 2.1 新增功能

#### SQLite数据库存储 (`src/evaluation/result_db.py`)
- **问题**: 多进程并发写入JSON文件导致数据覆盖丢失
- **解决**: SQLite + WAL模式，天然支持并发写入
- **Schema**:
  - `evaluation_results`: 单问题评估结果
  - `model_summaries`: 模型汇总指标

#### 增量保存
- 每个问题评估完成后立即写入数据库
- 避免长时间运行后崩溃导致数据丢失
- 支持实时监控进度

#### 新增命令行参数
| 参数 | 说明 |
|------|------|
| `--db <path>` | SQLite数据库路径 (默认: `outputs/results.db`) |
| `--export-json <path>` | 导出数据库为JSON |
| `--import-json <path>` | 从JSON导入到数据库 |
| `--workers <n>` | 样本级并发数 |
| `--resume` | 断点续传，跳过已完成问题 |

### 2.2 使用示例

```bash
# 评估单模型（结果实时写入SQLite）
python scripts/evaluation/evaluate_llm.py \
    --model gpt-5.2-chat \
    --limit 200 \
    --workers 4 \
    --db outputs/results.db

# 断点续传
python scripts/evaluation/evaluate_llm.py \
    --model gpt-5.2-chat \
    --limit 500 \
    --resume \
    --db outputs/results.db

# 导入已有JSON结果
python scripts/evaluation/evaluate_llm.py \
    --import-json outputs/llm_200.json \
    --db outputs/results.db

# 实时监控进度
sqlite3 outputs/results.db \
    "SELECT model_name, COUNT(*), ROUND(100.0*SUM(success)/COUNT(*),1) as rr
     FROM evaluation_results GROUP BY model_name"
```

---

## 三、200样本大规模评估（进行中）

### 3.1 配置
- **样本量**: 200
- **模型数**: 8个
- **max_steps**: 15
- **存储**: SQLite增量保存

### 3.2 当前进度

| Model | Completed | Success | RR |
|-------|-----------|---------|-----|
| gpt-5.2-chat | 200/200 | 199 | 99.5% |
| gpt-4.1-mini | 进行中 | - | ~56% |
| o4-mini | 进行中 | - | ~99% |
| gpt-5-mini | 进行中 | - | 100% |
| Llama-3.3-70B | 进行中 | - | ~95% |
| Kimi-K2-Thinking | 进行中 | - | 100% |
| o1 | 进行中 | - | 100% |
| gpt-5-nano | 进行中 | - | ~35% |

---

## 四、文档更新

### 4.1 CLAUDE.md更新
- 更新 `scripts/` 目录结构，包含完整文件列表
- 版本号: v0.4.0 → v0.5.0

### 4.2 新增进度文档
- `docs/progress/2026-01-12_scripts_reorganization.md` (本文档)

---

## 五、实验运行脚本

### 5.1 新增脚本

**位置**: `scripts/experiments/`

| 脚本 | 行数 | 用途 |
|------|------|------|
| `run_llm_eval.sh` | ~250行 | LLM评估主脚本 |
| `monitor_eval.sh` | ~100行 | 实时进度监控 |

### 5.2 使用示例

```bash
# 单模型评估
./scripts/experiments/run_llm_eval.sh --model gpt-5.2-chat --samples 200

# 全模型并行评估
./scripts/experiments/run_llm_eval.sh --all --samples 200 --parallel

# 断点续传
./scripts/experiments/run_llm_eval.sh --model o4-mini --samples 500 --resume

# 实时监控
./scripts/experiments/monitor_eval.sh --watch
```

### 5.3 自动生成的实验目录结构

```
outputs/experiments/2026-01-12/all_models_200samples/
├── config.yaml        # 实验配置（自动生成）
├── git_hash.txt       # 代码版本
├── results.db         # SQLite数据库
├── results.json       # 导出的JSON
└── logs/
    ├── gpt-5.2-chat.log
    ├── o4-mini.log
    └── ...
```

---

## 六、产出统计

| 类型 | 数量 |
|------|------|
| 新增文件 | 3 (`result_db.py`, `run_llm_eval.sh`, `monitor_eval.sh`) |
| 移动文件 | 14个脚本 |
| 修改文件 | 5 (`evaluate_llm.py`, `CLAUDE.md`, `__init__.py`, 进度日志等) |
| 新增代码 | ~550行 |

---

## 七、下一步

1. 等待8模型200样本评估完成
2. 汇总分析评估结果，生成论文级对比表
3. 考虑扩展到500样本进行统计显著性验证
4. 清理旧的临时输出文件

---

## 八、Test-Time Compute 追踪实现 ✅ 完成

### 8.1 背景

当前 `steps` 指标只捕捉外部环境交互次数，未追踪：
- Token 消耗（input/output/reasoning）
- API 调用次数
- 实际计算成本

这与文献中的 "Test-Time Compute" 定义不一致，导致推理模型 (o1) 的实际计算成本被低估。

### 8.2 实现内容

#### Phase 1: 数据结构扩展 (`src/evaluation/metrics.py`)

新增 `TokenUsage` 数据类：
```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0  # For o1/thinking models
    model: str = ""
    provider: str = ""
    timestamp: str = ""
```

扩展 `EpisodeResult` 新增字段：
- `total_tokens`, `total_input_tokens`, `total_output_tokens`, `total_reasoning_tokens`
- `api_call_count`, `tokens_per_step`, `wall_clock_seconds`

#### Phase 2: Agent Token 捕获 (`src/agents/llm_agent.py`)

- 新增 `_record_api_call()`, `get_token_stats()`, `reset_token_stats()` 方法
- 修改 `_call_openai()`, `_call_anthropic()`, `_call_azure_openai()`, `_call_azure_foundry()` 提取 token 使用
- 特别处理 o1/o4-mini 的 `reasoning_tokens` 字段

#### Phase 3: 流程集成 (`src/evaluation/benchmark_runner.py`, `episode_stats.py`)

- `BenchmarkRunner.run_episode()` 添加时间追踪和 token 收集
- `EpisodeTracker.finalize()` 接受 `token_stats` 和 `elapsed_seconds` 参数

#### Phase 4: 指标计算 (`src/evaluation/metrics.py`)

新增指标方法：
- `compute_avg_tokens()`: 平均 token 消耗
- `compute_tokens_per_step()`: 每步平均 token
- `compute_token_efficiency()`: Token 效率 = RR × 1000 / AvgTokens
- `compute_rr_at_token_budget(budget)`: 在给定 token 预算内的成功率

`compute_summary()` 新增输出：
- `avg_tokens`, `avg_input_tokens`, `avg_output_tokens`, `avg_reasoning_tokens`
- `tokens_per_step`, `token_efficiency`
- `rr_at_5k_tokens`, `rr_at_10k_tokens`, `rr_at_20k_tokens`, `rr_at_50k_tokens`

#### Phase 5: 数据库扩展 (`src/evaluation/result_db.py`)

- `evaluation_results` 表新增列：`total_tokens`, `input_tokens`, `output_tokens`, `reasoning_tokens`, `api_call_count`, `wall_clock_seconds`
- `model_summaries` 表新增列：`avg_tokens`, `tokens_per_step`, `token_efficiency` 等
- 添加 `_migrate_schema()` 支持旧数据库兼容

#### Phase 6: 输出显示 (`scripts/evaluation/evaluate_llm.py`)

更新 `print_comparison_table()` 显示 token 指标：
```
Agent                  RR      RR@5    Steps   Tokens    Tok/Step   Efficiency
----------------------------------------------------------------------------------------------------
gpt-5.2-chat           99.5%   78.0%     4.6     2,300      500.0       0.4326
o1                     94.0%   78.0%     4.0    12,000    3,000.0       0.0783
```

### 8.3 新增/修改文件

| 文件 | 修改类型 | 代码行数 |
|------|----------|----------|
| `src/evaluation/metrics.py` | 修改 | +120行 |
| `src/agents/llm_agent.py` | 修改 | +80行 |
| `src/evaluation/benchmark_runner.py` | 修改 | +15行 |
| `src/evaluation/episode_stats.py` | 修改 | +40行 |
| `src/evaluation/result_db.py` | 修改 | +60行 |
| `scripts/evaluation/evaluate_llm.py` | 修改 | +30行 |
| **总计** | | **~345行** |

### 8.4 使用示例

```bash
# 运行评估（自动追踪 token）
python scripts/evaluation/evaluate_llm.py \
    --model o4-mini \
    --limit 50 \
    --workers 4 \
    --db outputs/results.db

# 查看 token 统计
sqlite3 outputs/results.db "
    SELECT model_name,
           AVG(total_tokens) as avg_tokens,
           AVG(reasoning_tokens) as avg_reasoning,
           SUM(total_tokens) as total_cost
    FROM evaluation_results
    GROUP BY model_name
"
```

### 8.5 预期输出对比

实施后的对比表将显示：
- o1 模型虽然 `steps=4`，但 `tokens=12,000` (因 reasoning)
- gpt-5.2-chat `steps=4.6`，`tokens=2,300` (无 reasoning)
- Token Efficiency 使 o1 的真实计算成本可见

### 8.6 实现验证 ✅

**验证测试 (2026-01-12):**

```
Test 1 - TokenUsage dataclass: ✅
  input: 100, output: 50, total: 150, reasoning: 30

Test 2 - Agent token methods: ✅
  has reset_token_stats: True
  has get_token_stats: True
  has _record_api_call: True

Test 3 - EpisodeResult token fields: ✅
  total_tokens: 2500
  total_reasoning_tokens: 100
  api_call_count: 5
  wall_clock_seconds: 12.5

Test 4 - MetricsCalculator token methods: ✅
  avg_tokens: 3333
  tokens_per_step: 500.0
  token_efficiency: 0.2000
  rr_at_5k_tokens: 66.67%

Test 5 - Database schema: ✅
  evaluation_results: 6 token columns
  model_summaries: 8 token columns
```

**文件修改统计:**
- `src/evaluation/metrics.py`: +163 行 (TokenUsage, EpisodeResult扩展, 新指标方法)
- `src/agents/llm_agent.py`: +88 行 (token追踪逻辑)
- `src/evaluation/benchmark_runner.py`: +18 行 (时间和token收集)
- `src/evaluation/episode_stats.py`: +58 行 (token字段和finalize扩展)
- `src/evaluation/result_db.py`: +75 行 (schema扩展, 迁移逻辑)
- `scripts/evaluation/evaluate_llm.py`: +52 行 (输出表格更新)
- **总计: ~454 行**

---

*作者: Claude Code*
*日期: 2026-01-12*
