# LLM-for-OR: Agentic Operations Research with Large Language Models

**NeurIPS 2026 Research Project**

> From Static Translation to Dynamic Self-Correction in Operations Research

## Overview

This project develops evaluation benchmarks for LLM agents in Operations Research (OR) and Operations Management (OM). The core contribution is **OR-Debug-Bench**, a Gym-like MDP environment for solver infeasibility diagnosis and recovery.

### Research Directions

| Tier | Direction | Benchmark | Status |
|------|-----------|-----------|--------|
| 1 | A | OR-Debug-Bench | 🚧 In Development |
| 1 | B | OR-Bias-Bench | 📋 Planned |
| 1 | C | OR-Compliance-Bench | 📋 Planned |
| 2 | D-F | Formulation/Transfer/Disruption | 📋 Future |
| 3 | G-H | Safety-RL/Multi-Agent | 📋 Future |

## Key Features

- **MDP-based Evaluation**: Evaluate LLM agents as policies navigating solver states
- **Verifiable Rewards**: Solver status provides deterministic, noise-free feedback
- **Process Metrics**: Beyond outcome accuracy—measure diagnostic reasoning quality
- **Saboteur Agent**: Systematic error injection for benchmark data generation

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/LLM-for-OR.git
cd LLM-for-OR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Prerequisites

- Python 3.9+
- Gurobi Optimizer (academic license: free)
- CUDA (optional, for GRPO training)

## Project Structure

```
LLM-for-OR/
├── docs/                    # Research documentation
│   ├── 00_PROJECT_OVERVIEW.md
│   ├── directions/          # Research direction details
│   └── archive/             # Original reports
├── src/                     # Core code
│   ├── environments/        # MDP environments
│   ├── agents/              # LLM/RL agents
│   ├── solvers/             # Gurobi/Pyomo interfaces
│   ├── data_generation/     # Saboteur agent
│   └── evaluation/          # Metrics computation
├── configs/                 # Configuration files
├── data/                    # Datasets
├── experiments/             # Experiment configs
└── notebooks/               # Analysis notebooks
```

## Quick Start

```python
from src.environments import SolverGym
from src.agents import LLMAgent

# Create environment
env = SolverGym(solver="gurobi")

# Load agent
agent = LLMAgent(model="gpt-4o")

# Run episode
state = env.reset(problem="infeasible_tsp.mps")
done = False
while not done:
    action = agent.act(state)
    state, reward, done, info = env.step(action)

print(f"Recovery: {info['recovered']}, Steps: {info['steps']}")
```

## Documentation

- [Project Overview](docs/00_PROJECT_OVERVIEW.md)
- [Research Context](docs/01_RESEARCH_CONTEXT.md)
- [OR-Debug-Bench Details](docs/directions/A_OR_Debug_Bench/)
- [Literature Review](docs/03_LITERATURE_REVIEW.md)

## Citation

```bibtex
@article{llm-for-or-2026,
  title={OR-Debug-Bench: A Gym Environment for Solver Infeasibility Diagnosis and Recovery},
  author={Ao, Ruicheng},
  journal={NeurIPS},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Author**: Ruicheng Ao
- **Institution**: MIT IDSS
- **Email**: [your-email]

---

*Last updated: January 2026*
