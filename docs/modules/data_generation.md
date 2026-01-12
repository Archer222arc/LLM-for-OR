# Data Generation 模块接口文档

## 概述

**模块定位**: 数据生成模块，通过受控错误注入生成训练和测试数据。

**核心功能**:
- Saboteur Agent错误注入
- 四种错误类型分类（Type A-D）
- 注入结果验证
- Ground truth修复方案生成
- 数据集批量生成

**研究方向**: Direction A (OR-Debug-Bench)
**相关文档**: [A3_Data_Generation.md](../directions/A_OR_Debug_Bench/A3_Data_Generation.md)

**模块路径**: `src/data_generation/`

---

## 概述

数据生成模块通过向可行优化模型注入受控错误，生成训练和测试数据。核心是 **Saboteur Agent**，能够系统性地引入不可行性。

---

## 核心接口

### SaboteurAgent

**文件**: `src/data_generation/saboteur_agent.py`

错误注入代理。

```python
class SaboteurAgent:
    def __init__(self, solver: GurobiSolver, seed: Optional[int] = None):
        """初始化Saboteur Agent"""

    def inject_type_a(self) -> InjectionResult:
        """Type A: 翻转约束方向 (≤ ↔ ≥)"""

    def inject_type_b(self) -> InjectionResult:
        """Type B: 改变变量类型"""

    def inject_type_c(self) -> InjectionResult:
        """Type C: 移除约束表达式项"""

    def inject_type_d(self) -> InjectionResult:
        """Type D: 添加矛盾约束"""

    def inject_random_error(self) -> InjectionResult:
        """随机选择错误类型注入"""
```

**错误类型**:

| Type | 名称 | 注入方式 | 预期结果 |
|------|------|----------|----------|
| **A** | Bound Error | 翻转约束方向 (≤ ↔ ≥) | INFEASIBLE |
| **B** | Variable Error | 改变变量类型 (INTEGER ↔ CONTINUOUS) | SUBOPTIMAL |
| **C** | Logic Error | 移除约束表达式项 | INFEASIBLE/UNBOUNDED |
| **D** | Conflict Error | 添加矛盾约束 | INFEASIBLE |

---

## 使用模式

### 基本错误注入

```python
from src.solvers import GurobiSolver
from src.data_generation import SaboteurAgent

# 创建可行模型
solver = GurobiSolver.from_file("feasible_model.mps")
assert solver.solve().status == "OPTIMAL"

# 注入错误
saboteur = SaboteurAgent(solver, seed=42)
result = saboteur.inject_type_d()  # 添加冲突约束

print(f"Error type: {result.error_type}")
print(f"Target: {result.target_name}")
print(f"New status: {result.solver_status}")
print(f"Ground truth fix: {result.ground_truth_fix}")
```

### 批量数据生成

```python
from src.solvers import GurobiSolver
from src.data_generation import SaboteurAgent, ErrorType

# 加载多个可行模型
models = ["model1.mps", "model2.mps", "model3.mps"]

for model_file in models:
    solver = GurobiSolver.from_file(model_file)
    saboteur = SaboteurAgent(solver, seed=42)

    # 注入不同类型错误
    for error_type in [ErrorType.TYPE_A, ErrorType.TYPE_D]:
        result = saboteur.inject_error(error_type)
        if result.success:
            # 保存为训练数据
            save_problem(solver, result)
```

---

## 扩展指南

### 添加新错误类型

1. 在 `ErrorType` 枚举中添加新类型
2. 在 `SaboteurAgent` 中实现 `inject_type_X()` 方法
3. 更新 `inject_random_error()` 包含新类型
4. 添加对应单元测试

### 自定义注入策略

```python
class CustomSaboteur(SaboteurAgent):
    def inject_custom_error(self):
        # 自定义错误注入逻辑
        pass
```

---

## 测试策略

**文件**: `tests/unit/test_saboteur_agent.py`

**覆盖**:
- ErrorType枚举
- 各类型错误注入
- 注入结果验证
- Ground truth正确性

**运行**:
```bash
pytest tests/unit/test_saboteur_agent.py -v
```

---

## 依赖关系

### 内部依赖

- `src/solvers/`: GurobiSolver求解器

### 外部依赖

- `gurobipy`: Gurobi Python API

---

## 参考文献

- **Saboteur概念**: PILOT-Bench (ICLR 2026)
- **错误注入**: Controlled Error Injection for MIP Models

---

*最后更新: 2026-01-11*
