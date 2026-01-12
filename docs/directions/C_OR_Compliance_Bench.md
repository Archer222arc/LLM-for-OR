# Direction C: OR-Compliance-Bench — Hierarchical Rule Composition

## Core Concept

Evaluate LLM agents' ability to apply complex, hierarchical rule systems in supply chain compliance. HSCodeComp (ICLR 2026) found a massive gap: **46.8% agent vs 95% human accuracy** on HS code classification, specifically noting failures in combining multi-level rules.

---

## Why This is Novel

| Aspect | Justification |
|--------|---------------|
| **Literature Gap** | HSCodeComp is expert-level benchmark but focuses only on classification accuracy, not the reasoning process |
| **LLM-Specific** | Trade compliance requires understanding legal text, applying hierarchical exceptions, and generating audit trails—fundamentally NL tasks |
| **Practical Impact** | Trade compliance errors cost billions annually; automation is high-value |

---

## MDP Formulation

### State Space
- **Product description**: Natural language specification
- **Applicable regulations**: Multi-jurisdictional rules
- **Precedent database**: Historical classification decisions
- **Current classification attempt**: Partial decision tree

### Action Space
- Rule selection
- Exception application
- Classification decision
- Audit justification generation

### Reward Function
```
R = Classification_Accuracy + Explanation_Quality + Regulatory_Acceptance

Where:
- Classification_Accuracy: Match with expert classification
- Explanation_Quality: Audit trail completeness
- Regulatory_Acceptance: Would customs accept this reasoning?
```

---

## Proposed Benchmark: "OR-Compliance-Bench"

### Task Categories

| Task | Description | Complexity |
|------|-------------|------------|
| **Task 1: HS Code Classification** | 6-digit to 10-digit refinement | Multi-level hierarchy |
| **Task 2: Export Control Determination** | Dual-use goods, sanctions screening | Multi-jurisdictional |
| **Task 3: Customs Valuation** | Transfer pricing, royalty allocation | Calculation + rules |

### Metrics

| Metric | Definition |
|--------|------------|
| **Rule Composition Accuracy** | Correctly combine multiple rules |
| **Exception Handling Rate** | Properly apply exceptions to general rules |
| **Audit Trail Quality** | Explanation matches decision reasoning |

---

## Hierarchical Rule Structure

### Example: HS Code Classification

```
Level 1: Section (e.g., "Live animals; animal products")
    └── Level 2: Chapter (e.g., "Meat and edible meat offal")
        └── Level 3: Heading (e.g., "Meat of bovine animals")
            └── Level 4: Subheading (e.g., "Fresh or chilled")
                └── Level 5: Tariff item (e.g., "Carcasses and half-carcasses")

Challenge: Agent must navigate all 5 levels with exceptions at each
```

### Exception Handling Example

```
General Rule: "Electronic devices classified under Chapter 85"
Exception 1: "Unless primarily for medical use (Chapter 90)"
Exception 2: "Unless primarily for automotive use (Chapter 87)"
Sub-exception: "Medical devices for automotive safety → Chapter 87"

Agent must correctly apply nested exception logic
```

---

## Connection to PILOT-Bench

| PILOT-Bench | OR-Compliance-Bench | Comparison |
|-------------|---------------------|------------|
| Tool-driven workflows | Rule-driven workflows | Different verification type |
| Probabilistic tool outcomes | Deterministic rule application | Different uncertainty source |
| Multi-step reasoning | Hierarchical reasoning | Both test sequential decision |

---

## Data Requirements

| Requirement | Source |
|-------------|--------|
| HS Code database | World Customs Organization |
| Expert classifications | Trade compliance firms |
| Exception rules | Customs regulations |
| Audit trail examples | Regulatory filings |

---

## Evaluation Protocol

1. **Classification Accuracy**: Match with expert gold standard
2. **Reasoning Trace**: Evaluate step-by-step rule application
3. **Exception Detection**: Test on edge cases with exceptions
4. **Multi-jurisdiction**: Test cross-border rule conflicts

---

## Challenges and Considerations

| Challenge | Mitigation |
|-----------|------------|
| Expert knowledge required | Partner with trade compliance experts |
| Regulatory data access | Use public HS code databases |
| Jurisdiction complexity | Start with single jurisdiction (US/EU) |
| Explanation evaluation | Develop rubric with domain experts |

---

*Back to: [TIER1_Overview.md](TIER1_Overview.md)*
