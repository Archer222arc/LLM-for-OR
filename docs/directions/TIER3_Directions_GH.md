# TIER 3 Research Directions: G, H

> **TIER 3 Criteria:** Promising concepts with existing solutions in adjacent domains; may require more foundational work before benchmark creation.

---

## Direction G: Safety-Constrained RL for Operational Decisions

### Core Concept

Ensure LLM agents respect **hard operational constraints** during exploration and deployment. Causal-GNN SupplyNets uses Lyapunov-safe RL to guarantee work-in-process limits are satisfied with high probability.

### Challenge

Standard RL explores freely, potentially violating safety constraints. In operations, constraint violations can be **catastrophic**:
- Stockouts leading to lost customers
- Safety incidents in manufacturing
- Regulatory breaches with legal consequences

Need RL methods that guarantee constraint satisfaction during learning.

### Current State

| Approach | Status | Gap |
|----------|--------|-----|
| Lyapunov-safe RL | Exists for continuous control | Adaptation to LLM agents is open |
| Causal-GNN | Focus on supply chain WIP limits | Broader operational constraints unexplored |
| Constrained MDP | Theoretical foundations exist | Integration with LLM reasoning is novel |

### Research Questions

1. How to encode operational constraints in LLM action space?
2. Can we provide formal safety guarantees with LLM agents?
3. Trade-off between exploration and constraint satisfaction?

### Potential Directions

| Direction | Description |
|-----------|-------------|
| **Constraint-aware prompting** | Encode constraints in system prompt |
| **Safe action masking** | Filter unsafe actions before execution |
| **Lyapunov critic** | Train safety critic alongside policy |

---

## Direction H: Multi-Agent Coordination in Service Operations

### Core Concept

Evaluate coordination among **specialized LLM agents** managing different aspects of service operations. Hospital workflow research documents multi-agent systems where bed management, staff scheduling, and patient flow agents must negotiate.

### Challenge

Individual agents may optimize local objectives while degrading system performance. Need evaluation of:
- Emergent coordination quality
- Conflict resolution effectiveness
- System-level optimality vs. local optimality

### Multi-Agent Architecture

```
Service Operations System
├── Bed Management Agent
│   └── Objective: Maximize bed utilization
├── Staff Scheduling Agent
│   └── Objective: Minimize overtime
├── Patient Flow Agent
│   └── Objective: Minimize wait times
└── Coordination Layer
    └── Negotiate resource allocation
```

### Proposed Evaluation Dimensions

| Dimension | Metric |
|-----------|--------|
| **Coordination Quality** | Nash Equilibrium Distance (how far from stable coordination) |
| **Conflict Resolution Rate** | Successful negotiations / total conflicts |
| **System-Level Optimality** | Overall objective vs. sum of individual objectives |

### Application Domains

| Domain | Agents | Coordination Challenge |
|--------|--------|------------------------|
| **Hospital** | Beds, Staff, Patient Flow | Resource competition |
| **Warehouse** | Picking, Packing, Shipping | Throughput synchronization |
| **Contact Center** | Routing, Staffing, Queue | Real-time demand matching |

---

## TIER 3 Development Path

### Direction G: Safety-Constrained RL

```
Phase 1: Literature review on safe RL
Phase 2: Prototype constraint encoding for LLM agents
Phase 3: Benchmark design for safety evaluation
Phase 4: Integration with Direction A (safe debugging)
```

### Direction H: Multi-Agent Coordination

```
Phase 1: Survey multi-agent OR applications
Phase 2: Design coordination evaluation metrics
Phase 3: Prototype hospital/warehouse environment
Phase 4: Benchmark development
```

---

## Connection to TIER 1-2 Directions

| TIER 3 | Connects To | Synergy |
|--------|-------------|---------|
| G: Safety-Constrained | A: OR-Debug-Bench | Safe exploration during debugging |
| G: Safety-Constrained | B: OR-Bias-Bench | Constraint-aware bias mitigation |
| H: Multi-Agent | F: Disruption | Multi-agent disruption response |
| H: Multi-Agent | C: Compliance | Multi-jurisdiction coordination |

---

*See also: [SYNERGY_AND_PORTFOLIO.md](SYNERGY_AND_PORTFOLIO.md) for portfolio strategy*
