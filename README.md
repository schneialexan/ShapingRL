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
│   ├── core/                     # Core utilities (config, seeds, wandb)
│   ├── environments/             # Environment wrappers
│   │   ├── mountain_car.py      # MountainCar environment
│   │   ├── montezuma.py         # Montezuma base environment (make_montezuma)
│   │   ├── montezuma_rgb.py     # RGB state space training
│   │   ├── montezuma_ram.py     # RAM-only state space training
│   │   ├── montezuma_rgb_ram.py # RGB+RAM dictionary training
│   │   ├── montezuma_grey.py    # Grayscale training
│   │   └── montezuma_grey_ram.py # Grey+RAM training
│   ├── shaping/                  # Reward shaping methods
│   └── training/                 # Training utilities
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── flake.nix                    # Nix flake for reproducible dev environment
└── requirements.txt             # Pip requirements file
```

---

## 🎯 Usage

### MountainCar Environment

```bash
python src/environments/mountain_car.py --agent=ppo --shaping=potential --timesteps=10000
python src/environments/mountain_car.py --agent=dqn --shaping=naive --timesteps=5000
python src/environments/mountain_car.py --agent=q-learning --shaping=none --timesteps=2000
```

### Montezuma's Revenge Environments

Each environment file has its own wandb project and fixed state space configuration:

**RAM-only (recommended for speedrunning):**
```bash
python src/environments/montezuma_ram.py --timesteps 2500000 --shaping=manhattan --checkpoint-interval 500000
```

**RGB:**
```bash
python src/environments/montezuma_rgb.py --timesteps 1000000 --shaping=manhattan
```

**RGB+RAM (Dictionary, MultiInputPolicy):**
```bash
python src/environments/montezuma_rgb_ram.py --timesteps 1000000
```

**Grayscale:**
```bash
python src/environments/montezuma_grey.py --timesteps 1000000
```

**Grayscale+RAM:**
```bash
python src/environments/montezuma_grey_ram.py --timesteps 1000000
```

### Continuing Training from Checkpoint

```bash
python src/environments/montezuma_ram.py \
    --continue-from ./checkpoints/checkpoint_2500000_steps.zip \
    --timesteps 1000000 \
    --checkpoint-interval 500000
```

**Important:** Use the same environment file as the original training!

### Common CLI Options

All Montezuma environment scripts support:
- `--timesteps`: Number of training timesteps
- `--seed`: Random seed (default: 42)
- `--device`: cuda or cpu (default: cuda)
- `--shaping`: Reward shaping method (None, "manhattan")
- `--checkpoint-path`: Directory for checkpoints (default: ./checkpoints)
- `--checkpoint-interval`: Save checkpoint every N timesteps (default: 100000)
- `--continue-from`: Path to checkpoint file to continue training
- `--save-checkpoints`: Directory for saving during continued training
- `--no-wandb`: Disable wandb logging
- `--project`: WandB project name (default: auto-detected from script)

---

## 📦 Checkpointing

### What are Checkpoints?

Checkpoints are saved states of your trained model at specific points during training. They contain:
- **Model Weights**: Current neural network weights
- **Optimizer State**: Optimizer state (for continuous training)
- **Training Metadata**: Timesteps, hyperparameters, etc.

### What are VecNormalize Stats?

**VecNormalize** is a wrapper that normalizes observations and rewards to stabilize training. The "Stats" are:
- **Running Mean**: Average values of observations/rewards
- **Running Variance**: Variance of observations/rewards
- **Count**: Number of samples used for statistics

**Why are they important?**

When continuing from a checkpoint, VecNormalize stats **must** be loaded, otherwise:
- Normalization starts from scratch
- Model receives different normalized inputs
- Training can become unstable or performance drops

### Checkpoint Files

Stable-Baselines3 saves checkpoints as:
- **Model**: `checkpoint_XXXXXX_steps.zip` (contains model weights, optimizer state, metadata)
- **VecNormalize Stats**: `checkpoint_XXXXXX_steps_vecnormalize.pkl` (normalization statistics)

Example:
```
checkpoints/
├── checkpoint_1000000_steps.zip
├── checkpoint_1000000_steps_vecnormalize.pkl
├── checkpoint_2000000_steps.zip
└── checkpoint_2000000_steps_vecnormalize.pkl
```

### How Checkpointing Works

Checkpointing is **always enabled** when `--checkpoint-path` is provided. Checkpoints are automatically saved:
- Model: `checkpoint_XXXXXX_steps.zip`
- VecNormalize Stats: `checkpoint_XXXXXX_steps_vecnormalize.pkl`

The system automatically:
1. Saves checkpoints during training at specified intervals
2. Loads VecNormalize stats before loading the model (critical!)
3. Preserves timestep counter for continuous training

---

## 🧪 Development

### Running Tests

```bash
# All tests
pytest

# Specific test files
pytest tests/test_mc.py
pytest tests/test_mr.py
pytest tests/test_benchmark.py
pytest tests/test_dependencies.py

# Quick environment verification (all state spaces)
pytest tests/test_all_environments.py
```

### Code Quality

```bash
# Format code
ruff format .

# Run linter
ruff check .
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
- **opencv-python** - Image processing
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
