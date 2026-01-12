# Environments 模块接口文档

## 概述

**模块定位**: MDP环境实现，提供Gymnasium风格的强化学习接口，用于求解器不可行性诊断任务。

**核心功能**:
- 封装Gurobi求解器为MDP环境
- 定义State空间（DebugState）
- 定义Action空间（ActionType, Action）
- 实现Reward结构（三层奖励）
- 提供标准Gymnasium接口（reset/step）

**研究方向**: Direction A (OR-Debug-Bench)
**相关文档**: [A2_MDP_Spec.md](../directions/A_OR_Debug_Bench/A2_MDP_Spec.md)

**模块路径**: `src/environments/`

---

## 设计理念

### 1. Gymnasium兼容性

遵循OpenAI Gymnasium标准，提供统一的RL接口：
```python
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
```

### 2. 状态不可变性

`DebugState` 使用 dataclass 实现，每次状态转移返回新实例，避免副作用。

### 3. 三层奖励结构

- **Outcome Reward**: 终态奖励（+100 OPTIMAL, -50 INFEASIBLE）
- **Process Reward**: 过程奖励（+10 IIS缩小，-1 步数惩罚）
- **Faithfulness Penalty**: 一致性惩罚（-20 诊断矛盾）

### 4. Episode终止条件

- **Terminated**: 成功（OPTIMAL）或显式 SUBMIT
- **Truncated**: 达到 max_steps 上限
- **Done**: terminated or truncated

---

## 核心接口

### SolverDebugEnv

**文件**: `src/environments/solver_gym.py`

主环境类，封装求解器为MDP环境。

#### 构造函数

```python
def __init__(
    self,
    solver: GurobiSolver,
    problem_nl: str = "",
    max_steps: int = 50,
    reward_config: Optional[RewardConfig] = None,
    seed: Optional[int] = None,
)
```

**参数**:
- `solver`: GurobiSolver 实例
- `problem_nl`: 问题的自然语言描述（可选）
- `max_steps`: 最大步数限制（默认50）
- `reward_config`: 自定义奖励配置
- `seed`: 随机种子

#### 主要方法

##### reset()

```python
def reset(
    self,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
) -> Tuple[DebugState, dict]
```

重置环境到初始状态。

**返回**:
- `state`: 初始DebugState
- `info`: 元信息字典

**行为**:
- 重置求解器到原始模型
- 执行初始求解
- 计算IIS（如果不可行）
- 返回初始状态

##### step()

```python
def step(
    self, action: Action
) -> Tuple[DebugState, float, bool, bool, dict]
```

执行动作并返回下一状态。

**参数**:
- `action`: Action 对象

**返回** (Gymnasium标准):
- `state`: 新状态
- `reward`: 标量奖励
- `terminated`: 是否达到终态
- `truncated`: 是否超时
- `info`: 额外信息

**动作处理**:
1. 验证动作合法性
2. 应用动作到求解器
3. 重新求解
4. 计算奖励
5. 检查终止条件
6. 返回新状态

#### 属性

```python
@property
def state(self) -> DebugState:
    """当前状态"""

@property
def is_done(self) -> bool:
    """Episode是否结束"""

@property
def action_space(self) -> spaces.Discrete:
    """动作空间（Gym格式）"""

@property
def observation_space(self) -> spaces.Dict:
    """观察空间（Gym格式）"""
```

---

## 数据结构

### DebugState

**文件**: `src/environments/state.py`

MDP状态的完整表示。

```python
@dataclass(frozen=True)
class DebugState:
    solver_status: str                 # OPTIMAL/INFEASIBLE/UNBOUNDED
    constraint_names: List[str]        # 所有约束名称
    variable_names: List[str]          # 所有变量名称
    iis_constraints: List[str]         # IIS约束集
    iis_bounds: List[str]              # IIS边界集
    objective: Optional[float]         # 目标值
    step_count: int                    # 当前步数
    problem_nl: str = ""               # 问题描述
```

#### 辅助方法

```python
def is_optimal(self) -> bool:
    """是否为最优状态"""

def is_infeasible(self) -> bool:
    """是否不可行"""

def get_iis_size(self) -> int:
    """IIS大小（约束数+边界数）"""

def to_dict(self) -> dict:
    """序列化为字典"""
```

**设计特点**:
- `frozen=True`: 不可变，确保状态一致性
- 包含完整的求解器反馈信息
- 支持序列化和比较

---

### Action & ActionType

**文件**: `src/environments/action.py`

#### ActionType 枚举

```python
class ActionType(Enum):
    # 诊断动作
    GET_IIS = "get_iis"
    CHECK_SLACK = "check_slack"

    # 修复动作
    DROP_CONSTRAINT = "drop_constraint"
    RELAX_CONSTRAINT = "relax_constraint"
    UPDATE_RHS = "update_rhs"
    UPDATE_BOUNDS = "update_bounds"

    # 元动作
    RESET = "reset"
    SUBMIT = "submit"
```

**动作分类**:

| 类别 | 动作 | 需要参数 | 说明 |
|------|------|----------|------|
| **诊断** | GET_IIS | 无 | 计算IIS |
| **诊断** | CHECK_SLACK | target | 检查约束松弛值 |
| **修复** | DROP_CONSTRAINT | target | 删除约束 |
| **修复** | RELAX_CONSTRAINT | target, value | 放松约束 |
| **修复** | UPDATE_RHS | target, value | 更新右侧常数 |
| **修复** | UPDATE_BOUNDS | target, value, value2 | 更新变量边界 |
| **元** | RESET | 无 | 重置模型 |
| **元** | SUBMIT | 无 | 提交当前模型 |

#### Action 数据类

```python
@dataclass
class Action:
    action_type: ActionType
    target: Optional[str] = None      # 目标约束/变量名
    value: Optional[float] = None     # 值1
    value2: Optional[float] = None    # 值2（UPDATE_BOUNDS用）
```

#### 工厂函数

便捷的动作创建函数：

```python
def get_iis() -> Action
def check_slack(constraint: str) -> Action
def drop_constraint(constraint: str) -> Action
def relax_constraint(constraint: str, epsilon: float) -> Action
def update_rhs(constraint: str, new_rhs: float) -> Action
def update_bounds(variable: str, lb: float, ub: float) -> Action
def reset() -> Action
def submit() -> Action
```

---

### Reward & RewardCalculator

**文件**: `src/environments/reward.py`

#### RewardConfig

```python
@dataclass
class RewardConfig:
    # Outcome rewards
    optimal_reward: float = 100.0      # 成功恢复OPTIMAL
    infeasible_penalty: float = -50.0  # SUBMIT时仍INFEASIBLE

    # Process rewards
    iis_reduction_reward: float = 10.0 # 每减少1个IIS约束
    step_penalty: float = -1.0         # 每步惩罚

    # Faithfulness penalty
    diagnosis_error_penalty: float = -20.0  # 诊断矛盾
```

#### RewardCalculator

```python
class RewardCalculator:
    def __init__(self, config: RewardConfig = None)

    def calculate(
        self,
        prev_state: DebugState,
        action: Action,
        next_state: DebugState,
        terminated: bool,
    ) -> float
```

**奖励计算逻辑**:

1. **Outcome Reward** (仅在 terminated 时):
   - `next_state.is_optimal()` → +100
   - `action == SUBMIT and next_state.is_infeasible()` → -50

2. **Process Reward** (每步):
   - IIS约束减少 → `+10 × Δcount`
   - 步数惩罚 → `-1`

3. **Faithfulness Penalty** (未来功能):
   - 诊断与求解器日志矛盾 → `-20`

**示例**:
```python
# Step 1: GET_IIS
# 奖励: -1 (step penalty)

# Step 2: DROP_CONSTRAINT (移除IIS约束)
# 奖励: -1 (step) + 10 (IIS reduction) + 100 (optimal) = +109
```

---

## 使用模式

### 基本流程

```python
from src.solvers import GurobiSolver
from src.environments import SolverDebugEnv, Action, ActionType

# 1. 创建求解器
solver = GurobiSolver.from_file("model.mps")

# 2. 创建环境
env = SolverDebugEnv(solver, max_steps=50)

# 3. 重置环境
state, info = env.reset()
print(f"Initial status: {state.solver_status}")

# 4. 运行 episode
while not env.is_done:
    # 代理选择动作
    action = agent.act(state)

    # 执行动作
    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"Action: {action}, Reward: {reward}")

    state = next_state

    if terminated or truncated:
        break

print(f"Final status: {state.solver_status}")
```

### 自定义奖励

```python
from src.environments import RewardConfig

# 自定义奖励
config = RewardConfig(
    optimal_reward=200.0,       # 提高成功奖励
    step_penalty=-2.0,          # 增加步数惩罚
)

env = SolverDebugEnv(solver, reward_config=config)
```

### 手动调试

```python
from src.environments import get_iis, drop_constraint, submit

# Step 1: 获取IIS
action = get_iis()
state, reward, _, _, _ = env.step(action)

# Step 2: 删除IIS约束
if state.iis_constraints:
    target = state.iis_constraints[0]
    action = drop_constraint(target)
    state, reward, _, _, _ = env.step(action)

# Step 3: 提交
action = submit()
state, reward, terminated, _, _ = env.step(action)
```

---

## 扩展指南

### 添加新动作类型

1. 在 `ActionType` 枚举中添加新动作
2. 在 `Action` 数据类中添加必要参数
3. 在 `SolverDebugEnv.step()` 中实现动作逻辑
4. 更新奖励计算（如需要）
5. 添加工厂函数

### 自定义State表示

继承 `DebugState` 并添加新字段：

```python
@dataclass(frozen=True)
class ExtendedDebugState(DebugState):
    slack_values: Dict[str, float] = field(default_factory=dict)
    model_complexity: int = 0
```

### 实现新的Reward策略

继承 `RewardCalculator` 并重写 `calculate()`:

```python
class CustomRewardCalculator(RewardCalculator):
    def calculate(self, prev_state, action, next_state, terminated):
        # 自定义奖励逻辑
        return custom_reward
```

---

## 测试策略

### 单元测试

**文件**: `tests/unit/test_solver_gym.py`

**覆盖**:
- ActionType 枚举和 Action 创建
- DebugState 序列化和方法
- RewardCalculator 各类奖励
- SolverDebugEnv reset/step 流程
- Episode 终止条件

**运行**:
```bash
pytest tests/unit/test_solver_gym.py -v
```

### 集成测试

**测试场景**:
1. 简单不可行模型完整episode
2. 多步修复流程
3. 超时截断
4. 不同动作组合

---

## 依赖关系

### 内部依赖

- `src/solvers/`: 求解器接口（GurobiSolver）
- `src/agents/`: 代理使用环境

### 外部依赖

- `gymnasium`: RL环境标准
- `gurobipy`: Gurobi求解器（通过 solvers模块）

---

## 参考文献

- **Gymnasium文档**: https://gymnasium.farama.org/
- **PILOT-Bench**: Process-oriented reward shaping (ICLR 2026)
- **CorrectBench**: Self-correction in optimization (arXiv 2025)

---

*最后更新: 2026-01-11*
