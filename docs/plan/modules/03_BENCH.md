---
Research Direction: Direction A (OR-Debug-Bench)
Phase: Phase 1 - 数据构建
Last Updated: 2026-01-12
Related Code: src/data_generation/
Status: ✅ Robust注入方法已实现
---

# Phase 1: 数据构建 (Benchmark)

*返回: [项目计划](../PROJECT_PLAN.md)*

---

## 1.1 错误类型设计 (Novelty: 系统化错误分类)

**文献依据**: MOID (arXiv 2025) 验证LLM可以做infeasibility diagnosis

| Type | 名称 | 实现方法 | IIS特征 | 难度 |
|------|------|----------|---------|------|
| A | 方向翻转 | `<=` → `>=` | IIS含多个约束 | Medium |
| B | 变量类型 | Integer → Binary+tight bound | IIS含变量bound | Hard |
| C | 缺失项 | 删除约束中的term | 间接导致infeasibility | Hard |
| D | 直接矛盾 | 添加`x >= a AND x <= b, a>b` | IIS只含1-2约束 | Easy |

---

## 任务1.1.1: 修复Type A生成器 (方向翻转)

```python
# 当前问题: 翻转后可能仍feasible
# 根本原因: 随机选择约束，没有考虑约束的"紧度"

# ============================================
# 修复方案: 基于Slack的智能选择
# ============================================

def inject_type_a_robust(model):
    """
    Novelty体现: 生成的问题必须有"诊断价值"
    - 不是随便翻转一个约束
    - 而是翻转"紧约束"(slack接近0)
    - 这确保翻转后必然infeasible
    """
    # Step 1: 先求解原模型，获取slack信息
    solve(model)
    slack_info = {c.ConstrName: c.Slack for c in model.getConstrs()}

    # Step 2: 按slack排序，优先选择紧约束
    candidates = sorted(
        [c for c in model.getConstrs() if c.Sense != GRB.EQUAL],
        key=lambda c: abs(slack_info[c.ConstrName])
    )

    # Step 3: 迭代尝试，直到成功
    for c in candidates[:10]:  # 最多尝试10个
        model_copy = copy_model(model)
        flip_constraint(model_copy, c)

        if solve(model_copy).status == INFEASIBLE:
            iis = compute_iis(model_copy)
            if len(iis) >= 2:  # 确保IIS有意义，不是trivial
                return InjectionResult(
                    success=True,
                    error_type="A",
                    target=c.ConstrName,
                    iis=iis,
                    difficulty=classify_difficulty(iis)
                )

    return None  # 如果都失败，这个模型不适合Type A

def classify_difficulty(iis):
    """根据IIS大小分类难度"""
    if len(iis) <= 2:
        return "easy"
    elif len(iis) <= 5:
        return "medium"
    else:
        return "hard"
```

---

## 任务1.1.2: 修复Type B生成器 (变量类型)

```python
def inject_type_b_robust(model):
    """
    Type B的Novelty: 测试对变量语义的理解

    策略: 将INTEGER变量改为BINARY，同时收紧bound
    这创造了更subtle的infeasibility
    """
    int_vars = [v for v in model.getVars() if v.VType == GRB.INTEGER]

    for v in int_vars:
        model_copy = copy_model(model)
        var = model_copy.getVarByName(v.VarName)

        # 关键: 不只是改类型，还要收紧bound
        original_ub = var.UB
        var.VType = GRB.BINARY  # 0-1变量
        var.UB = 1
        var.LB = 0

        # 添加一个forcing constraint
        # 例如: x >= 2 (但x现在是binary，必然infeasible)
        if original_ub > 1:
            model_copy.addConstr(var >= 2, name=f"_force_{v.VarName}")
            model_copy.update()

            if solve(model_copy).status == INFEASIBLE:
                iis = compute_iis(model_copy)
                return InjectionResult(
                    success=True,
                    error_type="B",
                    target=v.VarName,
                    iis=iis,
                    metadata={"original_type": "INTEGER", "original_ub": original_ub}
                )

    return None
```

---

## 任务1.1.3: 修复Type C生成器 (缺失项)

```python
def inject_type_c_robust(model):
    """
    Type C的Novelty: 测试对约束结构的理解

    策略: 移除约束中的关键term，导致间接infeasibility
    """
    for c in model.getConstrs():
        row = model.getRow(c)
        if row.size() >= 2:  # 至少有2个term才能移除
            for i in range(row.size()):
                model_copy = copy_model(model)
                constr = model_copy.getConstrByName(c.ConstrName)
                var = row.getVar(i)
                coeff = row.getCoeff(i)

                # 将系数设为0 (相当于移除term)
                model_copy.chgCoeff(constr, model_copy.getVarByName(var.VarName), 0)
                model_copy.update()

                if solve(model_copy).status == INFEASIBLE:
                    iis = compute_iis(model_copy)
                    return InjectionResult(
                        success=True,
                        error_type="C",
                        target=f"{c.ConstrName}:{var.VarName}",
                        iis=iis,
                        metadata={"removed_term": var.VarName, "original_coeff": coeff}
                    )

    return None
```

---

## 难度分层标准

```
Easy:   IIS ≤ 2 constraints (Type D为主)
Medium: IIS 3-5 constraints (Type A为主)
Hard:   IIS ≥ 6 constraints (Type B/C为主)
```

---

## 1.2 数据源与规模

| 数据源 | 数量 | 用途 | 实现复杂度 |
|--------|------|------|------------|
| 合成LP/MIP | 5000 | 主要来源，可控难度 | Low |
| MIPLIB 2017 | 500 | 真实工业问题 | Medium (需解析MPS) |
| NL4Opt | 800 | 带NL描述 | Low |
| **总计** | **~6300** | train:val:test = 8:1:1 | - |

**注**: 原计划10K可能过于激进，6K是更现实的目标

---

## 1.3 数据验证Pipeline

```python
def validate_problem(original, sabotaged, error_info):
    """四重验证确保数据质量"""
    # 1. 原模型必须可解
    assert solve(original).status == OPTIMAL

    # 2. 破坏模型必须不可解
    assert solve(sabotaged).status == INFEASIBLE

    # 3. IIS必须包含target
    iis = compute_iis(sabotaged)
    assert error_info["target"] in iis

    # 4. 修复后必须恢复
    fixed = apply_ground_truth_fix(sabotaged, error_info)
    assert solve(fixed).status == OPTIMAL
```

---

*相关文档*:
- [MDP设计](01_MDP_DESIGN.md) - Bench数据结构由MDP State定义
- [连贯性设计](06_COHERENCE.md) - Bench如何指导Training
