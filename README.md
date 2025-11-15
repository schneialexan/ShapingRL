# 📈 ShapingRL

Reward Shaping and Potential-Based Methods in Reinforcement Learning

## 📚 Project Overview

**Course:** Reinforcement Learning  
**Topic:** Reward Shaping and Potential-Based Methods

### Objectives

Compare naïve dense shaping against potential-based shaping across sparse-reward tasks to identify when shaping accelerates learning versus when it biases policies.

### Tasks

- Implement standard RL algorithms (Q-learning, PPO, DQN) on sparse-reward tasks (MountainCar, Atari Montezuma's Revenge)
- Compare naive reward shaping (dense rewards, penalties) with potential-based shaping
- Study the impact on convergence, exploration, and policy optimality
- Analyze sensitivity to mis-specified shaping potentials
- Provide guidelines: When is shaping helpful, harmful, or unnecessary?

### References

1. [Ng, Andrew Y., Daishi Harada, and Stuart Russell. "Policy invariance under reward transformations: Theory and application to reward shaping." ICML 1999.](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
2. [Razin, Noam, et al. "What makes a reward model a good teacher? an optimization perspective." arXiv preprint arXiv:2503.15477 (2025).](https://arxiv.org/pdf/2503.15477)
3. [Ma, Haozhe, et al. "Reward shaping for reinforcement learning with an assistant reward agent." Forty-first international conference on machine learning. 2024.](https://dl.acm.org/doi/10.5555/3692070.3693450)

---

## 🏗️ Repository Structure

```text
ShapingRL/
├── src/                          # Source code
│   ├── agents/                   # RL algorithm implementations
│   │   ├── q_learning.py        # Tabular Q-Learning
│   │   ├── dqn.py               # Deep Q-Network
│   │   └── ppo.py               # Proximal Policy Optimization
│   ├── core/                     # Core utilities
│   │   ├── config.py            # Configuration settings
│   │   ├── logger.py            # Logging utilities
│   │   ├── seeds.py             # Seed management
│   │   └── wandb_callback.py   # Weights & Biases integration
│   ├── environments/             # Environment wrappers
│   │   ├── mountain_car.py      # MountainCar environment wrapper
│   │   └── montezumas_revenge.py # Montezuma's Revenge environment wrapper
│   ├── shaping/                  # Reward shaping methods
│   │   ├── mountain_car/        # MountainCar shaping methods
│   │   │   ├── naive.py         # Naive dense reward shaping
│   │   │   └── potential.py     # Potential-based reward shaping
│   │   └── montezuma/           # Montezuma's Revenge shaping methods
│   │       ├── naive.py         # Naive dense reward shaping
│   │       ├── potential.py     # Potential-based reward shaping
│   │       └── grid_based.py    # Grid-based shaping
│   ├── training/                 # Training utilities
│   │   ├── coach.py             # Guideline coach implementation
│   │   └── guide.py             # Guideline definitions
│   ├── info/                     # Information extraction
│   ├── information_extraction/   # Information extraction utilities
│   ├── log/                      # Logging utilities
│   └── util.py                   # General utilities
├── tests/                        # Unit tests
│   ├── test_benchmark.py        # Benchmark tests
│   ├── test_dependencies.py     # Dependency tests
│   ├── test_mc.py               # MountainCar tests
│   └── test_mr.py               # Montezuma's Revenge tests
├── notebooks/                    # Jupyter notebooks for experiments
│   └── exploratory_analysis.ipynb
├── flake.nix                    # Nix flake for reproducible dev environment
└── requirements.txt             # Pip requirements file
```

---

## 🎯 Usage

### Running Experiments

```bash
# MountainCar environment
python src/environments/mountain_car.py --agent=ppo --shaping=potential --timesteps=10000
python src/environments/mountain_car.py --agent=dqn --shaping=naive --timesteps=5000
python src/environments/mountain_car.py --agent=q-learning --shaping=none --timesteps=2000

# Montezuma's Revenge environment
python src/environments/montezumas_revenge.py --agent=ppo --shaping=potential --timesteps=100000
```

### Development

```bash
# Run tests
pytest

# Run specific test files
pytest tests/test_mc.py
pytest tests/test_mr.py
pytest tests/test_benchmark.py

# Format code
ruff format src/

# Run linter
ruff check src/
```

## Architecture

```mermaid
---
config:
  look: neo
---
flowchart TD
 subgraph Environments["Environments"]
        Env1["MountainCar-v0<br>(mountain_car.py)"]
        Env2@{ label: "Montezuma's Revenge<br>(montezumas_revenge.py)" }
  end
 subgraph subGraph1["Shaping Methods"]
        S1["Potential-Based<br>(potential_based.py)"]
        S2["Naive Dense<br>(naive_dense.py)"]
        S3["No Shaping"]
  end
 subgraph Agents["Agents"]
        A1["PPO<br>(ppo.py)"]
        A2["DQN<br>(dqn.py)"]
        A3["Q-Learning<br>(q_learning.py)"]
  end
 subgraph subGraph3["Available Components"]
    direction LR
        Environments
        subGraph1
        Agents
  end
 subgraph subGraph4["Execution Pipeline"]
        step1["Experiment Script selects Environment<br>(e.g. MountainCar)"]
        step2["Apply Shaping Method<br>(e.g. Potential-Based)"]
        step3["Initialize Agent<br>(e.g. PPO)"]
        step4["Train Agent on Wrapped Env"]
        step5["Log metrics to Weights &amp; Biases"]
  end
    step1 --> step2
    step2 --> step3
    step3 --> step4
    step4 --> step5
    Env2@{ shape: rect}
    style Env1 fill:#FDEDEC,stroke:#B03A2E,stroke-width:2px,color:#A93226
    style Env2 fill:#FDEDEC,stroke:#B03A2E,stroke-width:2px,color:#A93226
    style S1 fill:#D5F5E3,stroke:#239B56,stroke-width:2px,color:#1D8348
    style S2 fill:#D5F5E3,stroke:#239B56,stroke-width:2px,color:#1D8348
    style S3 fill:#D5F5E3,stroke:#239B56,stroke-width:2px,color:#1D8348
    style A1 fill:#D6EAF8,stroke:#2874A6,stroke-width:2px,color:#21618C
    style A2 fill:#D6EAF8,stroke:#2874A6,stroke-width:2px,color:#21618C
    style A3 fill:#D6EAF8,stroke:#2874A6,stroke-width:2px,color:#21618C
```

---

## 📦 Dependencies

### Core Dependencies

- **torch** - Deep learning framework (DQN, PPO)
- **gymnasium** - RL environments (MountainCar, Atari)
- **ale-py** - Atari Learning Environment
- **pygame** - Environment rendering
- **numpy** - Scientific computing
- **matplotlib** - Visualization
- **opencv-python** - Image processing (for Montezuma's Revenge test wrapper)
- **jupyter** - Interactive notebooks
- **wandb** - Experiment tracking and logging
- **stable-baselines3** - RL algorithm implementations

### Development Tools

- **pytest** - Testing framework
- **ruff** - Code formatting and linting

### Additional Dependencies

- **imageio**, **imageio-ffmpeg**, **moviepy**, **pillow** - Video recording for wandb

See `requirements.txt` or `flake.nix` for complete dependency list with versions.

---

## 👥 Team

- Björn Brauer
- Leandro Gregorini
- Flavio Kluser
- Alexandru Schneider
- Roman Zberg
