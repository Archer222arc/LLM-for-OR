# Evaluation 模块接口文档

## 概述

**模块定位**: 评估指标模块，提供全面的性能评估工具。

**核心功能**:
- Episode结果记录（EpisodeResult）
- 指标计算（MetricsCalculator）
- 基准测试运行（BenchmarkRunner）
- Episode追踪（EpisodeTracker）
- 轨迹聚合分析

**研究方向**: Direction A (OR-Debug-Bench)
**相关文档**: [A4_Evaluation_Metrics.md](../directions/A_OR_Debug_Bench/A4_Evaluation_Metrics.md)

**模块路径**: `src/evaluation/`

---

## 设计理念

### 1. 多层次评估

- **Episode级**: 单个episode的成功/失败
- **Trajectory级**: 动作序列分析
- **Benchmark级**: 多问题、多代理比较

### 2. 标准化指标

定义统一的评估指标（Recovery Rate, Diagnosis Accuracy等），便于跨研究比较。

### 3. 灵活配置

通过 `BenchmarkConfig` 支持不同评估设置。

---

## 核心接口

### MetricsCalculator

**文件**: `src/evaluation/metrics.py`

指标计算类。

```python
class MetricsCalculator:
    def compute_recovery_rate(
        self, results: List[EpisodeResult]
    ) -> float:
        """计算恢复率（成功率）"""

    def compute_avg_steps(
        self, results: List[EpisodeResult]
    ) -> float:
        """计算平均步数"""

    def compute_avg_reward(
        self, results: List[EpisodeResult]
    ) -> float:
        """计算平均奖励"""

    def compute_diagnosis_accuracy(
        self,
        results: List[EpisodeResult],
        ground_truth_map: Optional[Dict[str, List[str]]]
    ) -> Optional[float]:
        """计算诊断准确率"""

    def compute_summary(
        self,
        results: List[EpisodeResult],
        ground_truth_map: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """计算综合摘要"""

    def format_summary(self, summary: Dict[str, Any]) -> str:
        """格式化摘要为可读字符串"""
```

**关键指标**:

| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **Recovery Rate** | 成功恢复到OPTIMAL的比例 | success_count / total_episodes |
| **Diagnosis Accuracy** | IIS约束识别准确率 | correct_iis / total_iis |
| **Avg Steps** | 平均步数 | mean(steps) |
| **Avg Reward** | 平均总奖励 | mean(total_reward) |
| **Step Efficiency** | 奖励/步数比 | avg_reward / avg_steps |

### BenchmarkRunner

**文件**: `src/evaluation/benchmark_runner.py`

基准测试运行器。

```python
class BenchmarkRunner:
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """初始化Runner"""

    def run_episode(
        self, env, agent, problem_id: str = ""
    ) -> EpisodeResult:
        """运行单个episode"""

    def run_benchmark(
        self,
        problems: List[BenchmarkProblem],
        agent: BaseAgent,
    ) -> List[EpisodeResult]:
        """运行完整基准测试"""

    def compare_agents(
        self,
        problems: List[BenchmarkProblem],
        agents: List[BaseAgent],
    ) -> Dict[str, Dict[str, Any]]:
        """比较多个代理"""

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总"""

    def format_summary(self) -> str:
        """格式化汇总"""
```

### EpisodeTracker

**文件**: `src/evaluation/episode_stats.py`

Episode追踪器。

```python
class EpisodeTracker:
    def __init__(
        self,
        agent_name: str = "",
        problem_id: str = "",
        ground_truth_fix: Optional[str] = None
    ):
        """初始化Tracker"""

    def record_step(
        self,
        state: DebugState,
        action: Action,
        reward: float,
        next_state: DebugState,
        done: bool
    ) -> None:
        """记录单步"""

    def finalize(self, success: bool) -> EpisodeResult:
        """生成最终结果"""

    def reset(self) -> None:
        """重置Tracker"""
```

---

## 数据结构

### EpisodeResult

```python
@dataclass
class EpisodeResult:
    success: bool                           # 是否成功
    final_status: str                       # 最终状态
    steps: int                              # 总步数
    total_reward: float                     # 总奖励
    trajectory: List[Dict[str, Any]]        # 完整轨迹
    iis_actions: List[str]                  # IIS相关动作
    ground_truth_fix: Optional[str]         # Ground truth
    agent_name: str                         # 代理名称
    problem_id: str                         # 问题ID
```

### BenchmarkConfig

```python
@dataclass
class BenchmarkConfig:
    max_steps: int = 50           # 最大步数
    n_episodes: int = 1           # 每问题episode数
    seed: Optional[int] = None    # 随机种子
    verbose: bool = False         # 是否打印进度
```

---

## 使用模式

### 基本评估

```python
from src.evaluation import BenchmarkRunner, BenchmarkProblem

# 创建问题列表
problems = [
    BenchmarkProblem(problem_id="p1", env=env1),
    BenchmarkProblem(problem_id="p2", env=env2),
]

# 运行基准测试
runner = BenchmarkRunner()
results = runner.run_benchmark(problems, agent)

# 打印摘要
print(runner.format_summary())
```

### 代理比较

```python
from src.agents import HeuristicAgent, RandomAgent, GreedyDropAgent

agents = [
    HeuristicAgent(),
    RandomAgent(seed=42),
    GreedyDropAgent(),
]

comparison = runner.compare_agents(problems, agents)
print(runner.format_comparison(comparison))
```

### 自定义指标

```python
from src.evaluation import MetricsCalculator

calc = MetricsCalculator()

# 基本指标
recovery_rate = calc.compute_recovery_rate(results)
avg_steps = calc.compute_avg_steps(results)

# 诊断准确率（需要ground truth）
ground_truth = {
    "p1": ["c1", "c2"],
    "p2": ["c3"],
}
diagnosis_acc = calc.compute_diagnosis_accuracy(results, ground_truth)
```

---

## 扩展指南

### 添加新指标

```python
class CustomMetricsCalculator(MetricsCalculator):
    def compute_custom_metric(self, results):
        # 自定义指标计算
        pass
```

---

## 测试策略

**文件**: `tests/unit/test_evaluation.py`

**覆盖**:
- EpisodeResult数据类
- MetricsCalculator所有指标
- EpisodeTracker追踪逻辑
- BenchmarkRunner运行流程

**运行**:
```bash
pytest tests/unit/test_evaluation.py -v
# 31 passed in 0.47s
```

---

## 依赖关系

### 内部依赖

- `src/environments/`: DebugState, Action
- `src/agents/`: BaseAgent

---

## 参考文献

- **PILOT-Bench**: Process-oriented evaluation metrics (ICLR 2026)

---

*最后更新: 2026-01-11*
