#!/usr/bin/env python3
"""Test Montezuma's Revenge environment with all agent and shaping combinations."""

import os

import ale_py  # noqa: F401 - Required for ALE environment registration
import gymnasium as gym
import pytest
from gymnasium.envs.registration import register
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.dqn import DQNTrainer
from agents.ppo import PPOTrainer
from core.config import CONFIG
from core.seeds import reset_env_seed, set_seed
from shaping.montezuma.grid_based import GridBasedWrapper
from shaping.montezuma.naive import NaiveDenseWrapper
from shaping.montezuma.potential import MontezumaPotentialBasedWrapper
from training.coach import MontezumasKeyGuidelineCoachForcingOrder


def make_env(shaping="grid-based", render_mode=None):
    """Create Montezuma's Revenge environment with optional shaping."""
    # Check if rendering is enabled via environment variable
    if render_mode is None:
        render_mode = (
            "human"
            if os.environ.get("RENDER_TESTS", "").lower() in ("1", "true", "yes")
            else None
        )

    env_name = "ALE/MontezumaRevenge-v5"
    # Check if environment is already registered to avoid duplicate registration warning
    if env_name not in [spec.id for spec in gym.envs.registry.values()]:
        register(
            id=env_name,
            entry_point="ale_py.roms:MontezumaRevenge",
            max_episode_steps=4_000,
            reward_threshold=3000.0,
        )
    env = gym.make(env_name, render_mode=render_mode)

    if shaping == "naive":
        env = NaiveDenseWrapper(
            env, guideline_coach=MontezumasKeyGuidelineCoachForcingOrder
        )
    elif shaping == "potential":
        env = MontezumaPotentialBasedWrapper(env, gamma=0.99, debugging=False)
    elif shaping == "grid-based":
        env = GridBasedWrapper(env)

    return env


@pytest.mark.parametrize(
    "agent,shaping,device",
    [
        ("ppo", "grid-based", "cpu"),
        ("ppo", "naive", "cpu"),
        ("ppo", "potential", "cpu"),
        ("dqn", "grid-based", "cpu"),
        ("dqn", "naive", "cpu"),
        ("dqn", "potential", "cpu"),
        ("ppo", "grid-based", "cuda"),
        ("ppo", "naive", "cuda"),
        ("ppo", "potential", "cuda"),
        ("dqn", "grid-based", "cuda"),
        ("dqn", "naive", "cuda"),
        ("dqn", "potential", "cuda"),
        # Q-learning excluded: designed for low-dimensional observations (MountainCar), not image-based (Montezuma)
    ],
)
def test_montezuma(agent, shaping, device, timesteps=1000, seed=42):
    """Test a single agent/shaping combination."""
    render_enabled = os.environ.get("RENDER_TESTS", "").lower() in ("1", "true", "yes")
    # Note: Rendering will slow down tests significantly - use only for visualization
    render_info = " [RENDERING: ON (real-time)]" if render_enabled else ""
    print(f"Testing: agent={agent}, shaping={shaping}, device={device}{render_info}")

    # PPO with MLP-Policy should use CPU for image-based environments to avoid warnings
    if agent == "ppo":
        device = "cpu"

    # Set seed (may fall back to CPU if GPU unavailable)
    actual_device = set_seed(seed, device=device, deterministic_gpu=False)
    if actual_device != device:
        print(f"  ⚠ Warning: Requested {device}, but using {actual_device}")
        device = actual_device

    # Get config
    cfg = CONFIG.get(agent, {}).copy()
    cfg["total_timesteps"] = timesteps
    cfg["verbose"] = 0

    # Create environment (rendering controlled by RENDER_TESTS env var)
    env = make_env(shaping)
    reset_env_seed(env, seed)

    # Create trainer (run=None to disable wandb)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    if agent == "ppo":
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
        )
        trainer = PPOTrainer(vec_env, cfg, run=None, device=device, seed=seed)
    elif agent == "dqn":
        trainer = DQNTrainer(vec_env, cfg, run=None, device=device, seed=seed)

    # Train
    trainer.train()
    print(f"  ✓ Completed: agent={agent}, shaping={shaping}, device={device}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
