# 2026-01-11: Saboteur Agent 实现

**状态**: 完成

## 完成内容

### 1. `src/data_generation/error_types.py` (~95行)
- `ErrorType`: 错误类型枚举 (A/B/C/D)
- `InjectionResult`: 注入结果数据类
- 序列化方法 `to_dict()`

### 2. `src/data_generation/saboteur_agent.py` (~370行)
- `SaboteurAgent`: 核心错误注入类
- `inject_type_a()`: 约束方向翻转 (≤ ↔ ≥)
- `inject_type_b()`: 变量类型修改 (INTEGER ↔ CONTINUOUS)
- `inject_type_c()`: 表达式项移除 (系数置零)
- `inject_type_d()`: 矛盾约束添加 (保证INFEASIBLE)
- `inject_error()`: 按类型注入
- `inject_random_error()`: 随机注入
- `validate_injection()`: 验证注入效果
- `get_injection_history()`: 获取历史记录

### 3. `src/data_generation/__init__.py`
- 导出 `ErrorType`, `InjectionResult`, `SaboteurAgent`

### 4. `tests/unit/test_saboteur_agent.py` (~220行)
- 错误类型测试
- 四种注入类型测试
- 工具方法测试

## 验证结果

- ✓ Type A (约束翻转): 工作正常
- ✓ Type B (变量类型): 工作正常
- ✓ Type C (项移除): 工作正常
- ✓ Type D (矛盾约束): 保证INFEASIBLE

## 产出统计

| 文件 | 行数 | 说明 |
|------|------|------|
| error_types.py | 95 | 数据类定义 |
| saboteur_agent.py | 370 | 核心注入逻辑 |
| test_saboteur_agent.py | 220 | 单元测试 |
| **总计** | **685** | 数据生成模块核心 |

## 问题与解决

- **Gurobi addConstr API**: 初始实现使用错误的参数格式 `addConstr(expr, sense, rhs, name)`
- **解决**: 改为 `addConstr(expr >= rhs, name=name)` 格式

## 下一步

- [ ] 下载MIPLIB测试实例到 `data/raw/miplib/`
- [ ] 实现批量数据生成管道
- [ ] 实现 MDP 环境 `src/environments/solver_gym.py`
