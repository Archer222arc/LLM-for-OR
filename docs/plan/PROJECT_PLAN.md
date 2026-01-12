# OR-Debug-Bench 项目计划 v5.0

## 项目元信息

| 属性 | 值 |
|------|-----|
| **目标** | NeurIPS 2026投稿 - Direction A: OR-Debug-Bench |
| **硬件资源** | 2×A800/A100 (80GB VRAM) |
| **时间线** | Jan-May 2026 |

---

## 核心Novelty (4个)

| # | Novelty | 一句话描述 | 对标文献 |
|---|---------|----------|----------|
| 1 | **CorrectBench Gap** | OR领域首个self-correction评估基准 | CorrectBench |
| 2 | **MDP Policy** | 从Pass@1转向过程评估 π(aₜ\|sₜ) | StepORLM |
| 3 | **RLVR Oracle** | Solver作为verifiable reward | DeepSeek-R1 |
| 4 | **Test-time Compute** | RR@k衡量推理效率 | Scaling TTC |

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       OR-Debug-Bench Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Phase 1     │    │  Phase 2     │    │  Phase 3     │              │
│  │  数据构建    │───>│  评估框架    │───>│  RLVR训练    │              │
│  │  (2 weeks)   │    │  (2 weeks)   │    │  (4 weeks)   │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│        │                   │                   │                        │
│        v                   v                   v                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ 6K+问题      │    │ 6项指标      │    │ SFT→GRPO     │              │
│  │ 4错误类型    │    │ 11+模型评估  │    │ Qwen3-8B     │              │
│  │ 3难度等级    │    │ Test-time分析│    │ +10% RR      │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 模块文档索引

| 模块 | 内容 | 链接 |
|------|------|------|
| **MDP设计** | 形式化定义 + Bench/Eval/Training定位 | [01_MDP_DESIGN.md](modules/01_MDP_DESIGN.md) |
| **Novelty定位** | 4个Novelty的文献背景+贡献+实现指导 | [02_NOVELTY.md](modules/02_NOVELTY.md) |
| **Bench构建** | Phase 1: 错误类型 + 数据源 + 验证 | [03_BENCH.md](modules/03_BENCH.md) |
| **评估框架** | Phase 2: 指标实现 + 实验设计 | [04_EVAL.md](modules/04_EVAL.md) |
| **训练流程** | Phase 3: SFT/GRPO配置 + PRM扩展 | [05_TRAINING.md](modules/05_TRAINING.md) |
| **连贯性设计** | Bench→Eval→Training一致性 | [06_COHERENCE.md](modules/06_COHERENCE.md) |

---

## 详细时间线

| Week | Phase | 具体任务 | 交付物 |
|------|-------|----------|--------|
| 1 | Data | 修复Type A/B/C生成器 | 代码+测试 |
| 2 | Data | 构建6K数据集 | or_debug_bench_v1/ |
| 3 | Eval | 实现OP/FP/Faithfulness | metrics.py |
| 4 | Eval | 11模型完整评估 | results.json |
| 5 | Train | 收集SFT数据 | sft_train.json |
| 6 | Train | SFT训练+评估 | sft_checkpoint |
| 7 | Train | GRPO训练器实现 | grpo_trainer.py |
| 8 | Train | GRPO训练+消融 | grpo_checkpoint |
| 9-10 | Paper | 实验图表 | figures/ |
| 11-12 | Paper | 撰写+修改 | draft.pdf |

---

## 成功标准

| 指标 | 目标 | 验证方法 |
|------|------|----------|
| 数据规模 | ≥5000问题 | 统计 |
| SFT提升 | RR+5% vs base | 评估 |
| GRPO提升 | RR+5% vs SFT | 评估 |
| 论文投稿 | NeurIPS 2026 deadline | 提交 |

---

## 关键文献对照

| 我们的贡献 | 对标文献 | 我们的优势 |
|------------|----------|------------|
| OR Self-Correction | CorrectBench | 首个OR领域 |
| MDP评估 | SWE-bench | Solver-backed ground truth |
| GRPO训练 | DeepSeek-R1 | Domain-specific application |
| Process Reward | StepORLM | 扩展到debugging phase |
| Test-time Compute | Scaling TTC | OR领域首次验证 |

---

*最后更新: 2026-01-11*
*计划版本: v5.0 (模块化重构)*
