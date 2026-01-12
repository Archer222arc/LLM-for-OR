# 2026-01-12: Robust注入方法实现

**状态**: 完成

---

## 完成内容

### 1. 数据结构扩展 (`src/data_generation/error_types.py`)

- 新增 `Difficulty` 枚举类 (easy/medium/hard)
- 添加 `from_iis_size()` 类方法自动分类难度
- `InjectionResult` 扩展字段:
  - `difficulty`: 难度分类
  - `iis_size`: IIS约束数量
  - `iis_constraints`: IIS约束名列表
  - `iis_bounds`: IIS变量bound列表
  - `original_objective`: 原目标值（用于OP计算）

### 2. Robust注入方法 (`src/data_generation/saboteur_agent.py`)

| 方法 | 策略 | 预期成功率 |
|------|------|------------|
| `inject_type_a_robust()` | Slack-based智能约束选择 | 70-85% |
| `inject_type_b_robust()` | Forcing constraint策略 | 80-90% |
| `inject_type_c_robust()` | 对偶值敏感性分析 | 50-70% |
| `inject_type_d_robust()` | 链式冲突IIS控制 | 95%+ |

辅助方法:
- `inject_error_robust()`: 统一调用接口
- `inject_random_error_robust()`: 随机robust注入
- `_compute_iis_info()`: IIS信息提取
- `_inject_simple_conflict()`: 简单冲突（Easy难度）
- `_inject_chain_conflict()`: 链式冲突（Medium/Hard难度）

### 3. 四重验证模块 (`src/data_generation/validator.py`)

新建模块实现完整验证pipeline:

| Phase | 验证内容 | 失败处理 |
|-------|----------|----------|
| Phase 1 | 原模型必须OPTIMAL | 拒绝该问题 |
| Phase 2 | 破坏模型必须INFEASIBLE | 注入失败 |
| Phase 3 | IIS必须包含target | 注入无效 |
| Phase 4 | 修复后恢复OPTIMAL | 修复描述错误 |

核心类:
- `ProblemValidator`: 验证器主类
- `ValidationResult`: 验证结果数据类
- `validate_dataset()`: 批量验证函数

### 4. 生成脚本更新 (`scripts/generate_dataset.py`)

- 默认使用robust方法 (`--use_robust` / `--no_robust`)
- Type D按概率分配目标IIS大小 (30% easy, 45% medium, 25% hard)
- 新增统计:
  - 各Type成功率
  - 难度分布
  - IIS大小统计

---

## 产出统计

| 文件 | 修改行数 | 说明 |
|------|----------|------|
| `src/data_generation/error_types.py` | +40行 | Difficulty类 + 字段扩展 |
| `src/data_generation/saboteur_agent.py` | +400行 | 4个robust方法 + 辅助方法 |
| `src/data_generation/validator.py` | +350行 | 新建四重验证模块 |
| `src/data_generation/__init__.py` | +10行 | 导出新类 |
| `scripts/generate_dataset.py` | +80行 | robust支持 + 统计报告 |

**总计**: ~880行新增/修改代码

---

## 技术亮点

### Slack-based约束选择 (Type A)
```python
# 按slack排序，优先选择紧约束
candidates = sorted(constrs, key=lambda c: abs(c.Slack))
for target in candidates[:10]:
    # 翻转紧约束更可能导致infeasibility
```

### Forcing Constraint (Type B)
```python
# 变量改为binary后添加不可能满足的约束
var.VType = GRB.BINARY
var.UB = 1
model.addConstr(var >= 2, name=f"_force_{var.VarName}")  # 必然infeasible
```

### 对偶值敏感性 (Type C)
```python
# 高对偶值 = 紧约束，删除其中的term更可能破坏可行性
binding = [(c, abs(c.Pi)) for c in constrs if abs(c.Pi) > 1e-6]
binding.sort(key=lambda x: x[1], reverse=True)
```

### 链式冲突 (Type D)
```python
# 创建约束链形成循环冲突
# y >= x + 10, z >= y + 10, ..., x >= last + 10 (循环)
```

---

## 使用示例

```bash
# 使用robust方法生成50个问题
python scripts/generate_dataset.py \
    --n_problems 50 \
    --output data/benchmarks/or_debug_bench_v3

# 不使用robust方法（对比测试）
python scripts/generate_dataset.py \
    --n_problems 50 \
    --no_robust \
    --output data/benchmarks/baseline
```

---

## 下一步

1. ~~**运行测试验证**~~ ✅ 完成
2. ~~**大规模生成**~~ 进行中 (1000问题数据集)
3. **真实数据集成** - MIPLIB/NL4Opt集成
4. **评估实验** - 用新数据集评估LLM

---

## 更新 (2026-01-12 00:45)

### 验证结果

创建验证脚本 `scripts/validate_robust_methods.py`，测试结果：

| Type | Basic | Robust | 预期 | 状态 |
|------|-------|--------|------|------|
| A | 30% | 85% | >=70% | ✅ PASS |
| B | 0% | 100% | >=80% | ✅ PASS |
| C | 0% | 60% | >=50% | ✅ PASS |
| D | 100% | 100% | >=95% | ✅ PASS |

### Type C修复

原Type C方法成功率0%。问题：移除约束term通常不会导致infeasibility。

**解决方案**: 添加Strategy 4（Guaranteed Fallback）
```python
# 1. 创建紧约束对
upper: var <= opt_value + 0.5
lower: coeff * var >= lower_rhs  (coeff=1.0)

# 2. 修改系数导致冲突
coeff: 1.0 → 0.1
# 现在: 0.1 * var >= lower_rhs
# 即: var >= 10 * lower_rhs
# 与 var <= opt_value + 0.5 冲突!
```

### 模型生成器更新

更新 `scripts/generate_dataset.py` 中的模型生成器：
- `create_simple_lp()`: 添加混合约束(<=和>=) + 关键约束对
- `create_simple_mip()`: 同上，整数变量上限改为15

### 数据生成测试结果

**测试集** (100问题):
- 总成功率: 88%
- Type A: 80%, Type B: 100%, Type C: 72%, Type D: 100%
- 注: Type C在MIP上失败（Gurobi不支持MIP的dual值）

**生成中** (1000问题):
- 输出路径: `data/benchmarks/or_debug_bench_v1/`

---

*关联文档*:
- [03_BENCH.md](../plan/modules/03_BENCH.md) - 数据构建计划
- [06_COHERENCE.md](../plan/modules/06_COHERENCE.md) - Bench→Training连贯性
