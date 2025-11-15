#!/usr/bin/env python3
"""Test MountainCar environment with all agent and shaping combinations."""

import os
import sys

import pytest

# Add src folder to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

import gymnasium as gym  # noqa: E402
from gymnasium.wrappers import RecordEpisodeStatistics  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from agents.dqn import DQNTrainer  # noqa: E402
from agents.ppo import PPOTrainer  # noqa: E402
from agents.q_learning import QLearningTrainer  # noqa: E402
from core.config import CONFIG  # noqa: E402
from core.seeds import reset_env_seed, set_seed  # noqa: E402
from shaping.mountain_car.naive import NaiveDenseWrapper  # noqa: E402
from shaping.mountain_car.potential import PotentialBasedWrapper  # noqa: E402


def make_env(shaping="none", render_mode=None):
    """Create and configure the MountainCar environment."""
    # Check if rendering is enabled via environment variable
    if render_mode is None:
        render_mode = (
            "human"
            if os.environ.get("RENDER_TESTS", "").lower() in ("1", "true", "yes")
            else None
        )

    env = gym.make("MountainCar-v0", render_mode=render_mode)
    env = RecordEpisodeStatistics(env)
    env = Monitor(env)
    if shaping == "naive":
        env = NaiveDenseWrapper(env)
    elif shaping == "potential":
        env = PotentialBasedWrapper(env)
    return env


@pytest.mark.parametrize(
    "agent,shaping",
    [
        ("ppo", "none"),
        ("ppo", "naive"),
        ("ppo", "potential"),
        ("dqn", "none"),
        ("dqn", "naive"),
        ("dqn", "potential"),
        ("q-learning", "none"),
        ("q-learning", "naive"),
        ("q-learning", "potential"),
    ],
)
def test_mountain_car(agent, shaping, timesteps=1000, seed=42):
    """Test a single agent/shaping combination."""
    render_enabled = os.environ.get("RENDER_TESTS", "").lower() in ("1", "true", "yes")
    # Note: MountainCar doesn't support frameskip/speedup - rendering runs at real-time speed
    print(
        f"Testing: agent={agent}, shaping={shaping}"
        + (" [RENDERING: ON (real-time)]" if render_enabled else "")
    )

    # Set seed
    set_seed(seed, device="cpu", deterministic_gpu=False)

    # Get config
    cfg = CONFIG.get(agent, {}).copy()
    if "total_timesteps" in cfg:
        cfg["total_timesteps"] = timesteps
    elif agent == "q-learning":
        cfg["total_episodes"] = timesteps
    cfg["verbose"] = 0

    # Create environment (rendering controlled by RENDER_TESTS env var)
    env = make_env(shaping)
    reset_env_seed(env, seed)

    # Create trainer (run=None to disable wandb)
    if agent == "q-learning":
        trainer = QLearningTrainer(env, cfg, run=None, seed=seed)
    else:
        vec_env = DummyVecEnv([lambda: env])
        if agent == "ppo":
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
            )
        if agent == "ppo":
            trainer = PPOTrainer(vec_env, cfg, run=None, device="cpu", seed=seed)
        elif agent == "dqn":
            trainer = DQNTrainer(vec_env, cfg, run=None, device="cpu", seed=seed)

    # Train
    trainer.train()
    print(f"  ✓ Completed: agent={agent}, shaping={shaping}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
