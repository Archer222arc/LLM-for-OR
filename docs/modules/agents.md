# Agents 模块接口文档

## 概述

**模块定位**: 代理实现，提供多种策略用于MDP环境交互，包括LLM代理和基线代理。

**核心功能**:
- 定义代理抽象接口（BaseAgent）
- 实现LLM代理（支持OpenAI、Anthropic、Azure OpenAI、Azure Foundry）
- 提供基线代理（Random/Heuristic/Greedy/DoNothing）
- Prompt工程和状态格式化
- Episode历史追踪

**支持的LLM提供商**:
- `openai`: 直接OpenAI API
- `anthropic`: 直接Anthropic API
- `azure_openai`: Azure OpenAI Service (推荐用于生产环境)
- `azure_foundry`: Azure AI Foundry MaaS

**研究方向**: Direction A (OR-Debug-Bench)
**相关文档**: [A2_MDP_Spec.md](../directions/A_OR_Debug_Bench/A2_MDP_Spec.md)

**模块路径**: `src/agents/`

---

## 设计理念

### 1. 统一抽象接口

所有代理继承 `BaseAgent`，实现 `act()` 和 `reset()` 方法，确保互换性。

### 2. Provider无关设计

LLMAgent 支持多个LLM提供商（OpenAI, Anthropic），统一的API调用接口。

### 3. Prompt Engineering

- 系统提示定义代理角色和任务
- 状态格式化转换为LLM可读文本
- JSON响应格式确保可解析性

### 4. 可复现性

- RandomAgent 支持种子设置
- LLMAgent temperature=0 确保确定性输出

---

## 核心接口

### BaseAgent (抽象基类)

**文件**: `src/agents/base_agent.py`

所有代理必须继承的抽象基类。

```python
class BaseAgent(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._episode_history = []

    @abstractmethod
    def act(self, state: DebugState) -> Action:
        """根据当前状态选择动作"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置代理状态"""
        pass

    def record_step(
        self, state, action, reward, next_state, done
    ) -> None:
        """记录步骤用于学习（可选）"""

    def get_episode_history(self) -> List[dict]:
        """获取episode历史"""

    def clear_history(self) -> None:
        """清空历史"""
```

**核心方法**:
- `act()`: 策略核心，输入状态返回动作
- `reset()`: Episode间重置内部状态
- `record_step()`: 记录轨迹（用于离线学习）

---

### LLMAgent

**文件**: `src/agents/llm_agent.py`

基于大语言模型的调试代理。

#### 构造函数

```python
def __init__(
    self,
    model: str = "gpt-4",
    provider: str = "openai",
    temperature: float = 0.0,
    max_retries: int = 3,
    name: Optional[str] = None,
    # Azure OpenAI specific
    azure_endpoint: Optional[str] = None,
    api_version: str = "2024-10-21",
    azure_deployment: Optional[str] = None,
    # Azure Foundry specific
    foundry_endpoint: Optional[str] = None,
    foundry_model_id: Optional[str] = None,
    # Local config support
    use_local_config: bool = True,
    config_path: Optional[str] = None,
)
```

**参数**:
- `model`: 模型名称（例如 "gpt-4.1", "deepseek-r1"）
- `provider`: LLM提供商（"openai", "anthropic", "azure_openai", "azure_foundry"）
- `temperature`: 采样温度（0.0 = 确定性）
- `max_retries`: 解析失败重试次数
- `name`: 代理名称（默认"LLMAgent-{model}"）
- `use_local_config`: 是否使用本地配置文件（推荐True）
- `config_path`: 自定义配置文件路径

#### 使用示例

##### Azure OpenAI Service (推荐)

```python
from src.agents import LLMAgent

# 使用本地配置文件 (推荐)
agent = LLMAgent(
    model="gpt-4.1",
    provider="azure_openai",
    temperature=0.0,
    use_local_config=True,
)

action = agent.act(state)
```

##### Azure AI Foundry

```python
agent = LLMAgent(
    model="deepseek-r1",
    provider="azure_foundry",
    temperature=0.0,
    use_local_config=True,
)

action = agent.act(state)
```

##### OpenAI (直接API)

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-..."

agent = LLMAgent(
    model="gpt-4",
    provider="openai",
    temperature=0.0,
)

action = agent.act(state)
```

##### Anthropic Claude (直接API)

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

agent = LLMAgent(
    model="claude-3-sonnet-20240229",
    provider="anthropic",
    temperature=0.0,
)

action = agent.act(state)
```

#### 可用模型 (2026-01-11)

**Azure OpenAI Service**:
- GPT系列: gpt-4.1, gpt-4.1-mini, gpt-5-mini, gpt-5-nano, gpt-5.2-chat
- O系列: o1, o4-mini
- DeepSeek: DeepSeek-R1-0528, DeepSeek-V3.2
- 其他: Kimi-K2-Thinking, Llama-3.3-70B-Instruct

**Azure AI Foundry**:
- deepseek-r1

#### 内部方法

```python
def _build_prompt(self, state: DebugState) -> str:
    """构建LLM提示"""

def _call_llm(self, user_message: str) -> str:
    """调用LLM API"""

def _parse_response(self, response: str) -> Action:
    """解析JSON响应为Action"""
```

**工作流程**:
1. `format_state()` 将 DebugState 转为可读文本
2. 构建包含系统提示和用户消息的prompt
3. 调用LLM API获取响应
4. 从响应中提取JSON并解析为Action
5. 如果解析失败，重试最多max_retries次

---

### 基线代理

#### RandomAgent

**文件**: `src/agents/baseline_agents.py`

随机选择有效动作（下界基线）。

```python
class RandomAgent(BaseAgent):
    def __init__(self, seed: Optional[int] = None, name: str = "RandomAgent"):
        ...

    def act(self, state: DebugState) -> Action:
        valid_actions = self._get_valid_actions(state)
        return self._rng.choice(valid_actions)
```

**特点**:
- 从有效动作集合中均匀随机选择
- 支持种子设置保证可复现性
- 用作性能下界

#### HeuristicAgent

规则驱动的启发式代理（中等基线）。

```python
class HeuristicAgent(BaseAgent):
    def act(self, state: DebugState) -> Action:
        if state.is_optimal():
            return submit()

        if state.is_infeasible() and not state.iis_constraints:
            return get_iis()

        if state.iis_constraints:
            return drop_constraint(state.iis_constraints[0])

        return submit()
```

**策略**:
1. 如果OPTIMAL → SUBMIT
2. 如果INFEASIBLE且无IIS → GET_IIS
3. 如果有IIS → DROP第一个IIS约束
4. 其他 → SUBMIT

#### GreedyDropAgent

贪婪删除所有IIS约束。

```python
class GreedyDropAgent(BaseAgent):
    def act(self, state: DebugState) -> Action:
        if state.is_optimal():
            return submit()

        if not state.iis_constraints:
            return get_iis()

        # 删除所有未删除的IIS约束
        for constr in state.iis_constraints:
            if constr not in self._dropped_constraints:
                self._dropped_constraints.append(constr)
                return drop_constraint(constr)

        return submit()
```

**特点**:
- 更激进的修复策略
- 快速收敛但可能过度修改

#### DoNothingAgent

什么都不做，直接提交（失败基线）。

```python
class DoNothingAgent(BaseAgent):
    def act(self, state: DebugState) -> Action:
        return submit()
```

**用途**:
- 作为最低性能基线
- 验证环境正确性

---

## Prompt工程

**文件**: `src/agents/prompts.py`

### SYSTEM_PROMPT

系统提示定义代理角色和任务。

```python
SYSTEM_PROMPT = """
You are an expert Operations Research debugger. Your task is to diagnose
and repair infeasible optimization models.

## Available Actions:
- GET_IIS: Compute the Irreducible Infeasible Subsystem
- CHECK_SLACK(constraint): Check slack value for a constraint
- DROP_CONSTRAINT(constraint): Remove a constraint from the model
...

## Response Format:
Always respond with a JSON object:
{
    "reasoning": "Brief explanation",
    "action": "ACTION_NAME",
    "target": "constraint_name (if needed)",
    "value": 0.0 (if needed)
}
"""
```

**关键要素**:
- 角色定义：OR调试专家
- 动作列表：所有可用动作及参数
- 响应格式：JSON结构要求
- 示例：帮助LLM理解任务

### format_state()

将DebugState转换为LLM可读文本。

```python
def format_state(state: DebugState) -> str:
    return f"""
## Current State
- Status: {state.solver_status}
- Step: {state.step_count}
- IIS Constraints: {state.iis_constraints}
- IIS Bounds: {state.iis_bounds}

## Model Info
- Constraints: {len(state.constraint_names)}
- Variables: {len(state.variable_names)}

## History
{format_history(state.history)}

What action should be taken next?
"""
```

**设计考虑**:
- 清晰的层次结构
- 突出关键信息（Status, IIS）
- 包含历史上下文
- 明确的行动召唤

### format_history()

格式化动作历史。

```python
def format_history(history: List[dict]) -> str:
    if not history:
        return "No previous actions taken."

    lines = ["Previous Actions:"]
    for i, step in enumerate(history, 1):
        action = step["action"]
        result = step.get("result", {})
        lines.append(f"Step {i}: {action['action_type']}")
    return "\n".join(lines)
```

---

## 使用模式

### 基本使用

```python
from src.environments import SolverDebugEnv
from src.agents import HeuristicAgent

# 创建代理
agent = HeuristicAgent()

# 运行episode
state, _ = env.reset()
while not env.is_done:
    action = agent.act(state)
    state, reward, _, _, _ = env.step(action)

print(f"Final status: {state.solver_status}")
```

### 代理比较

```python
from src.agents import RandomAgent, HeuristicAgent, GreedyDropAgent
from src.evaluation import BenchmarkRunner, BenchmarkProblem

agents = [
    RandomAgent(seed=42),
    HeuristicAgent(),
    GreedyDropAgent(),
]

runner = BenchmarkRunner()
comparison = runner.compare_agents(problems, agents)
print(runner.format_comparison(comparison))
```

### LLM代理调试

```python
import logging

# 启用日志
logging.basicConfig(level=logging.DEBUG)

agent = LLMAgent(model="gpt-4", provider="openai")

# 查看prompt
state = env.reset()[0]
prompt = agent._build_prompt(state)
print("Prompt sent to LLM:")
print(prompt)

# 查看响应
action = agent.act(state)
print(f"Parsed action: {action}")
```

---

## 扩展指南

### 实现新代理

继承 `BaseAgent` 并实现必要方法：

```python
class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MyAgent")
        self._policy = load_policy()  # 自定义策略

    def act(self, state: DebugState) -> Action:
        # 实现决策逻辑
        features = self._extract_features(state)
        action = self._policy(features)
        return action

    def reset(self) -> None:
        # 重置内部状态
        self.clear_history()
```

### 添加新LLM Provider

在 `LLMAgent._call_llm()` 中添加新provider分支：

```python
def _call_llm(self, user_message: str) -> str:
    if self.provider == "openai":
        return self._call_openai(user_message)
    elif self.provider == "anthropic":
        return self._call_anthropic(user_message)
    elif self.provider == "your_provider":
        return self._call_your_provider(user_message)
    else:
        raise ValueError(f"Unknown provider: {self.provider}")
```

### 自定义Prompt

覆盖prompt模板：

```python
CUSTOM_PROMPT = """
[Your custom system prompt]
"""

agent = LLMAgent(model="gpt-4")
agent._system_prompt = CUSTOM_PROMPT  # 覆盖默认
```

---

## 测试策略

### 单元测试

**文件**: `tests/unit/test_agents.py`

**覆盖**:
- BaseAgent 抽象类验证
- 各基线代理行为正确性
- RandomAgent 可复现性
- HeuristicAgent 规则逻辑
- MockLLMAgent 预定义响应
- Prompt格式化函数

**运行**:
```bash
pytest tests/unit/test_agents.py -v
# 30 passed in 0.49s
```

### 集成测试

**测试场景**:
1. 代理与环境完整交互
2. Episode完整运行
3. 步骤记录和历史追踪

---

## 依赖关系

### 内部依赖

- `src/environments/`: DebugState, Action依赖
- `src/evaluation/`: 评估使用代理

### 外部依赖

- `openai`: OpenAI API（可选）
- `anthropic`: Anthropic API（可选）

**安装**:
```bash
pip install openai anthropic  # LLMAgent需要
```

---

## 参考文献

- **DeepSeek-R1**: GRPO算法实现 (Nature 2025)
- **PILOT-Bench**: 过程导向代理评估 (ICLR 2026)
- **ReAct**: Reasoning and Acting framework (ICLR 2023)

---

*最后更新: 2026-01-11*
