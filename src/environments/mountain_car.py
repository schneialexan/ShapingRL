import argparse
import os
import sys

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import wandb

# Add parent folder to sys.path (so imports work)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn import DQNTrainer
from agents.ppo import PPOTrainer
from agents.q_learning import QLearningTrainer
from core.config import CONFIG
from core.seeds import reset_env_seed, set_seed
from shaping.mountain_car.naive import NaiveDenseWrapper
from shaping.mountain_car.potential import PotentialBasedWrapper


def parse_args():
    """
    Parse command-line arguments using argparse (Python standard library).

    Usage examples:
        # With defaults (all parameters use defaults)
        python mountain_car.py

        # Using --key=value format (recommended)
        python mountain_car.py --agent=ppo --shaping=none --timesteps=2000 --run-name='test-run1' --wandb=false

        # Using --key value format (alternative)
        python mountain_car.py --agent ppo --shaping none --timesteps 2000 --run-name 'test-run1' --wandb false

        # Mixing formats
        python mountain_car.py --agent=ppo --shaping none --timesteps=2000

        # Using GPU
        python mountain_car.py --device=cuda --agent=dqn --timesteps=5000

        # Get help
        python mountain_car.py --help

    Note: argparse automatically handles:
    - Type conversion (e.g., timesteps is converted to int)
    - Validation (choices are enforced)
    - Help messages (--help flag)
    - Default values (shown in help)
    """
    parser = argparse.ArgumentParser(
        description="Train RL agents on MountainCar environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows defaults in help
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for training",
    )

    parser.add_argument(
        "--agent",
        choices=["ppo", "dqn", "q-learning"],
        default="ppo",
        help="RL agent to use for training",
    )

    parser.add_argument(
        "--shaping",
        choices=["none", "naive", "potential"],
        default="none",
        help="Reward shaping method to apply",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Number of training timesteps (small number for quick testing)",
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for the wandb run (default: auto-generated from parameters)",
    )

    parser.add_argument("--project", default="Mountain Car", help="WandB project name")

    def str_to_bool(value):
        """Convert string to boolean, accepting true/false/1/0 (case-insensitive)."""
        # Handle various input types for flexibility
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        value_lower = str(value).lower()
        if value_lower in ("true", "1", "yes", "on"):
            return True
        elif value_lower in ("false", "0", "no", "off"):
            return False
        else:
            raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

    parser.add_argument(
        "--wandb",
        type=str_to_bool,
        default=True,
        help="Enable (true/1) or disable (false/0) wandb logging",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--deterministic-gpu",
        type=str_to_bool,
        default=False,
        help="Enable deterministic GPU operations (slower but reproducible). Only affects GPU training. Default: False (optimize for speed)",
    )

    args = parser.parse_args()

    return args


def make_env(shaping="none"):
    """Create and configure the MountainCar environment."""
    env = gym.make("MountainCar-v0")
    # RecordEpisodeStatistics gives ep_rew_mean and ep_len_mean
    env = RecordEpisodeStatistics(env)
    # Stable Baselines3 Monitor wrapper
    env = Monitor(env)
    # Apply shaping wrapper
    if shaping == "naive":
        env = NaiveDenseWrapper(env)
    elif shaping == "potential":
        env = PotentialBasedWrapper(env)
    return env


def main():
    """Main training function."""
    args = parse_args()

    # Set all random seeds for reproducibility (may fall back to CPU if GPU unavailable)
    actual_device = set_seed(
        args.seed, device=args.device, deterministic_gpu=args.deterministic_gpu
    )
    if actual_device != args.device:
        # Update device if GPU was requested but unavailable
        args.device = actual_device

    # Get config for the selected agent (copy to avoid mutating global config)
    cfg = CONFIG.get(args.agent, {}).copy()
    if "total_timesteps" in cfg:
        cfg["total_timesteps"] = args.timesteps
    elif args.agent == "q-learning":
        # Q-learning uses episodes instead of timesteps
        cfg["total_episodes"] = args.timesteps

    # Set run name: auto-generate from parameters if not provided
    if args.run_name is None:
        run_name = f"{args.agent}_{args.shaping}"
    else:
        run_name = args.run_name

    # Run configuration is available via args if needed

    # Prepare tags for wandb (based on parameters)
    # Note: environment is already clear from the wandb project name
    tags = [args.device, args.agent, args.shaping]

    # Initialize Weights & Biases run (optional, allows training without wandb)
    # The run object is passed to trainers for logging (None if wandb disabled)
    run = None
    if args.wandb:
        run = wandb.init(
            project=args.project,
            name=run_name,
            tags=tags,
            config={
                "agent": args.agent,
                "shaping": args.shaping,
                "device": args.device,
                "timesteps": args.timesteps,
                "seed": args.seed,
                "deterministic_gpu": args.deterministic_gpu,
                **cfg,
            },
        )

    # Environment setup
    env = make_env(args.shaping)
    # Set environment seed
    reset_env_seed(env, args.seed)

    # Wrap in VecEnv for SB3 (not needed for q-learning which works directly with env)
    if args.agent == "q-learning":
        trainer = QLearningTrainer(env, cfg, run, seed=args.seed)
    else:
        vec_env = DummyVecEnv([lambda: env])
        if args.agent == "ppo":
            # Normalize observations/rewards for PPO stability (MountainCar has bounded obs)
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
            )
            trainer = PPOTrainer(vec_env, cfg, run, device=args.device, seed=args.seed)
        elif args.agent == "dqn":
            trainer = DQNTrainer(vec_env, cfg, run, device=args.device, seed=args.seed)

    trainer.train()

    # Only if wandb was activated, call wandb.finish()
    # This properly closes the wandb run
    if run:
        wandb.finish()


if __name__ == "__main__":
    main()
