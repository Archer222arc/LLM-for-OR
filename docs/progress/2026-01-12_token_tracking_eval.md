# Progress: Token Tracking Evaluation

**Date**: 2026-01-12
**Status**: Completed

---

## Summary

Implemented Test-Time Compute tracking and completed comprehensive evaluation of 8 LLM models on OR-Debug-Bench with 200 samples each. Generated publication-quality visualizations with all labels in English.

---

## Completed Tasks

### 1. Token Tracking Implementation
- Extended `EpisodeResult` with token fields (input/output/reasoning tokens)
- Modified `LLMAgent` to capture API usage from all providers
- Updated `ResultDB` schema with token columns
- Added metrics: Token Efficiency, RR@TokenBudget

### 2. Parallel Model Evaluation
- 8 models evaluated concurrently with separate SQLite databases
- 200 samples per model (1,600 total evaluations)
- Models: gpt-5.2-chat, o4-mini, o1, Kimi-K2-Thinking, gpt-5-mini, Llama-3.3-70B-Instruct, gpt-4.1-mini, gpt-5-nano

### 3. Visualization
- Created `scripts/visualization/plot_token_results.py`
- Generated 6 publication-quality figures (300 DPI, English labels)
- Figures: performance overview, token efficiency, RR@Steps, RR@TokenBudget, token composition, efficiency heatmap

### 4. Documentation
- Updated CLAUDE.md with visualization guidelines
- Generated COMPREHENSIVE_REPORT.md with all metrics

---

## Key Results

| Model | RR (%) | Avg Tokens | Efficiency | Rank |
|-------|--------|------------|------------|------|
| gpt-5.2-chat | 99.0 | 6,002 | 0.165 | 1 |
| o4-mini | 98.5 | 7,263 | 0.136 | 2 |
| o1 | 99.5 | 7,352 | 0.135 | 3 |
| Kimi-K2-Thinking | 100.0 | 8,640 | 0.116 | 4 |
| gpt-5-mini | 99.5 | 10,756 | 0.093 | 5 |
| Llama-3.3-70B-Instruct | 99.0 | 13,631 | 0.073 | 6 |
| gpt-4.1-mini | 59.0 | 27,094 | 0.022 | 7 |
| gpt-5-nano | 47.0 | 57,063 | 0.008 | 8 |

### Key Insights
1. **Token efficiency varies 20x** across models (0.008 to 0.165)
2. **gpt-5.2-chat** is optimal for cost-performance balance
3. **Reasoning models** (o1, o4-mini) achieve high RR@5 but with reasoning token overhead
4. **gpt-5-nano** appears stuck in reasoning loops (90% reasoning token ratio, 57k avg tokens)
5. **Failed attempts cost 2-7x more tokens** than successful ones

---

## Output Files

```
outputs/experiments/2026-01-12_token_tracking/
├── *.db (8 SQLite databases, one per model)
├── logs/*.log (8 log files)
├── figures/
│   ├── fig1_performance_overview.png
│   ├── fig2_token_efficiency.png
│   ├── fig3_rr_at_steps.png
│   ├── fig4_rr_at_tokens.png
│   ├── fig5_token_composition.png
│   └── fig6_efficiency_heatmap.png
└── COMPREHENSIVE_REPORT.md
```

---

## Code Changes

| File | Change |
|------|--------|
| `src/evaluation/metrics.py` | Added TokenUsage dataclass, extended EpisodeResult |
| `src/agents/llm_agent.py` | Added token tracking for all providers |
| `src/evaluation/result_db.py` | Extended schema with token columns |
| `scripts/visualization/plot_token_results.py` | New visualization script |
| `.claude/CLAUDE.md` | Added visualization guidelines |

---

## Next Steps

- [ ] Analyze per-problem difficulty correlation with token usage
- [ ] Investigate gpt-5-nano reasoning loops
- [ ] Run evaluation on error type subsets
- [ ] Prepare figures for paper submission

---

*Author: Claude Opus 4.5*
*Duration: ~4 hours (evaluation) + 1 hour (visualization)*
