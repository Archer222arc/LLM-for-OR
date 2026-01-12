# Solvers 模块接口文档

## 概述

**模块定位**: 求解器接口封装，提供统一的API用于与优化求解器（特别是Gurobi）交互。

**核心功能**:
- 封装Gurobi求解器为统一接口
- 提取求解器状态（OPTIMAL/INFEASIBLE/UNBOUNDED）
- 计算和解析IIS（Irreducible Infeasible Subsystem）
- 提供模型操作（克隆、重置、修改）
- 定义求解器相关数据结构

**研究方向**: Direction A (OR-Debug-Bench)
**相关文档**: [A_OR_Debug_Bench](../directions/A_OR_Debug_Bench/)

**模块路径**: `src/solvers/`

---

## 设计理念

### 1. 抽象与封装

`BaseSolver` 定义统一接口，隐藏求解器实现细节，便于支持多种求解器（Gurobi/CPLEX/Pyomo）。

### 2. 状态标准化

将不同求解器的状态码映射为统一的字符串表示（"OPTIMAL", "INFEASIBLE" 等），简化上层逻辑。

### 3. IIS提取核心

IIS（Irreducible Infeasible Subsystem）是诊断不可行性的关键，模块提供简洁的IIS提取接口。

### 4. 非破坏性操作

提供模型克隆和重置功能，确保原始模型不被修改。

---

## 核心接口

### GurobiSolver

**文件**: `src/solvers/gurobi_interface.py`

Gurobi求解器封装实现。

#### 构造与工厂方法

```python
class GurobiSolver:
    def __init__(self):
        """直接构造（通常不推荐）"""

    @classmethod
    def from_model(cls, model: gp.Model) -> "GurobiSolver":
        """从Gurobi Model对象创建"""

    @classmethod
    def from_file(cls, file_path: str) -> "GurobiSolver":
        """从文件（.mps/.lp）创建"""
```

**使用**:
```python
import gurobipy as gp
from src.solvers import GurobiSolver

# 方法1: 从Gurobi Model
m = gp.Model("my_model")
# ... 添加变量和约束 ...
solver = GurobiSolver.from_model(m)

# 方法2: 从文件
solver = GurobiSolver.from_file("model.mps")
```

#### 求解方法

```python
def solve(self) -> SolverState:
    """求解模型并返回状态"""
```

**返回**: `SolverState` 数据类，包含：
- `status`: 求解状态字符串
- `objective`: 目标值（如果有）
- `gap`: MIP gap（如果有）
- `solve_time`: 求解时间

**示例**:
```python
state = solver.solve()
print(f"Status: {state.status}")
if state.status == "OPTIMAL":
    print(f"Objective: {state.objective}")
```

#### IIS计算

```python
def compute_iis(self) -> IISResult:
    """计算Irreducible Infeasible Subsystem"""
```

**返回**: `IISResult` 数据类，包含：
- `constraints`: IIS约束名称列表
- `bounds`: IIS边界名称列表
- `size`: IIS总大小

**示例**:
```python
state = solver.solve()
if state.status == "INFEASIBLE":
    iis = solver.compute_iis()
    print(f"IIS constraints: {iis.constraints}")
    print(f"IIS bounds: {iis.bounds}")
    print(f"IIS size: {iis.size}")
```

#### 信息查询

```python
def get_constraint_info(self, name: str) -> ConstraintInfo:
    """获取约束详细信息"""

def get_variable_info(self, name: str) -> VariableInfo:
    """获取变量详细信息"""

def get_all_constraints(self) -> List[str]:
    """获取所有约束名称"""

def get_all_variables(self) -> List[str]:
    """获取所有变量名称"""
```

#### 模型操作

```python
def clone(self) -> "GurobiSolver":
    """克隆当前模型"""

def reset(self) -> None:
    """重置模型到原始状态"""

def relax_constraint(self, name: str, epsilon: float) -> None:
    """放松约束"""

def drop_constraint(self, name: str) -> None:
    """删除约束"""

def update_rhs(self, name: str, new_rhs: float) -> None:
    """更新约束右侧值"""

def update_bounds(self, name: str, lb: float, ub: float) -> None:
    """更新变量边界"""
```

---

## 数据结构

### SolverState

```python
@dataclass
class SolverState:
    status: str                    # "OPTIMAL", "INFEASIBLE", ...
    objective: Optional[float]     # 目标值
    gap: Optional[float]           # MIP gap
    solve_time: float              # 求解时间（秒）
```

### IISResult

```python
@dataclass
class IISResult:
    constraints: List[str]         # IIS约束名称
    bounds: List[str]              # IIS边界变量名称
    size: int                      # IIS大小
```

### ConstraintInfo

```python
@dataclass
class ConstraintInfo:
    name: str                      # 约束名称
    sense: str                     # "<=", ">=", "="
    rhs: float                     # 右侧值
    slack: Optional[float]         # Slack值
    is_in_iis: bool                # 是否在IIS中
```

### VariableInfo

```python
@dataclass
class VariableInfo:
    name: str                      # 变量名称
    vtype: str                     # "CONTINUOUS", "BINARY", "INTEGER"
    lb: float                      # 下界
    ub: float                      # 上界
    value: Optional[float]         # 当前值
    is_lb_in_iis: bool             # 下界是否在IIS中
    is_ub_in_iis: bool             # 上界是否在IIS中
```

---

## 使用模式

### 基本求解流程

```python
from src.solvers import GurobiSolver

# 创建求解器
solver = GurobiSolver.from_file("model.mps")

# 求解
state = solver.solve()

# 根据状态处理
if state.status == "OPTIMAL":
    print(f"Optimal value: {state.objective}")
elif state.status == "INFEASIBLE":
    iis = solver.compute_iis()
    print(f"Model is infeasible, IIS: {iis.constraints}")
```

### 模型修改与重新求解

```python
# 原始求解
state = solver.solve()
print(f"Original: {state.status}")

# 删除约束
solver.drop_constraint("problematic_constraint")

# 重新求解
new_state = solver.solve()
print(f"After repair: {new_state.status}")

# 恢复原始模型
solver.reset()
```

### 克隆模型

```python
# 克隆用于实验
cloned_solver = solver.clone()

# 在克隆上实验
cloned_solver.drop_constraint("c1")
state = cloned_solver.solve()

# 原始模型不受影响
original_state = solver.solve()
```

---

## 扩展指南

### 添加新求解器

实现 `BaseSolver` 接口：

```python
from src.solvers import BaseSolver

class CPLEXSolver(BaseSolver):
    def solve(self) -> SolverState:
        # CPLEX求解逻辑
        pass

    def compute_iis(self) -> IISResult:
        # CPLEX IIS计算
        pass

    # ... 实现其他方法
```

### 自定义状态码映射

```python
# 在GurobiSolver中覆盖状态映射
CUSTOM_STATUS_MAP = {
    gp.GRB.OPTIMAL: "SUCCESS",
    gp.GRB.INFEASIBLE: "FAIL",
    # ...
}
```

---

## 测试策略

**文件**: `tests/unit/test_gurobi_interface.py`

**覆盖**:
- 从文件/模型创建求解器
- 求解和状态提取
- IIS计算
- 模型克隆和重置
- 约束修改操作

**运行**:
```bash
pytest tests/unit/test_gurobi_interface.py -v
```

---

## 依赖关系

### 外部依赖

- `gurobipy`: Gurobi Python API（必须）

**安装**:
```bash
# Gurobi需要单独安装和许可证
pip install gurobipy
```

### 许可证

Gurobi需要有效许可证。学术用户可申请免费许可证：https://www.gurobi.com/academia/academic-program-and-licenses/

---

## 参考文献

- **Gurobi文档**: https://www.gurobi.com/documentation/
- **IIS概念**: Gurobi IIS Computation Guide

---

*最后更新: 2026-01-11*
