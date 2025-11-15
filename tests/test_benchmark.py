#!/usr/bin/env python3
"""Benchmark script to compare GPU vs CPU performance for PPO and DQN agents."""

import os
import sys
import time

import ale_py  # noqa: F401 - Required for ALE environment registration
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add src folder to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from agents.dqn import DQNTrainer  # noqa: E402
from agents.ppo import PPOTrainer  # noqa: E402
from core.config import CONFIG  # noqa: E402
from core.seeds import reset_env_seed, set_seed  # noqa: E402


def make_mountain_car_env():
    """Create and configure the MountainCar environment."""
    env = gym.make("MountainCar-v0")
    env = RecordEpisodeStatistics(env)
    env = Monitor(env)
    return env


def make_montezuma_env():
    """Create and configure the Montezuma's Revenge environment."""
    env_name = "ALE/MontezumaRevenge-v5"
    # Check if environment is already registered to avoid duplicate registration warning
    if env_name not in [spec.id for spec in gym.envs.registry.values()]:
        register(
            id=env_name,
            entry_point="ale_py.roms:MontezumaRevenge",
            max_episode_steps=4_000,
            reward_threshold=3000.0,
        )
    env = gym.make(env_name, render_mode=None)
    env = Monitor(env)
    return env


def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s"


def run_test(agent_type, device, timesteps=50000, seed=42, env_type="mountain_car"):
    """Run a single test and return the time taken."""
    env_name = "MountainCar" if env_type == "mountain_car" else "Montezuma"
    print(f"\n{'─' * 70}")
    print(
        f"  Testing {agent_type.upper():3s} on {device.upper():4s} ({env_name}, {timesteps:,} steps)"
    )
    print(f"{'─' * 70}")

    # Set seed
    actual_device = set_seed(seed, device=device, deterministic_gpu=False)
    if actual_device != device:
        print(f"  ⚠ Warning: Requested {device}, but using {actual_device}")
        device = actual_device

    # Get config
    cfg = CONFIG.get(agent_type, {}).copy()
    cfg["total_timesteps"] = timesteps
    cfg["verbose"] = 0  # Suppress output

    # Create environment
    if env_type == "mountain_car":
        env = make_mountain_car_env()
    else:  # montezuma
        env = make_montezuma_env()
    reset_env_seed(env, seed)

    # Wrap in VecEnv
    vec_env = DummyVecEnv([lambda: env])
    if agent_type == "ppo":
        if env_type == "mountain_car":
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        else:  # montezuma
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
            )

    # Create trainer (run=None to disable wandb)
    if agent_type == "ppo":
        trainer = PPOTrainer(vec_env, cfg, run=None, device=device, seed=seed)
    elif agent_type == "dqn":
        trainer = DQNTrainer(vec_env, cfg, run=None, device=device, seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Measure time
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    print(f"  ✓ Completed in {format_time(elapsed_time)}")

    return elapsed_time


def main():
    """Run all performance tests."""
    mountain_car_timesteps = 50_000
    montezuma_timesteps = 10_000
    seed = 42

    print("\n" + "=" * 70)
    print(" " * 20 + "GPU vs CPU Performance Benchmark")
    print("=" * 70)
    print(f"  MountainCar timesteps: {mountain_car_timesteps:,}")
    print(f"  Montezuma timesteps: {montezuma_timesteps:,}")
    print(f"  Seed: {seed}")
    print("=" * 70)

    results = {}

    # MountainCar Tests
    print("\n" + "=" * 70)
    print(" " * 25 + "MountainCar Benchmarks")
    print("=" * 70)

    results["mc_ppo_cpu"] = run_test("ppo", "cpu", mountain_car_timesteps, seed, "mountain_car")
    results["mc_ppo_gpu"] = run_test("ppo", "cuda", mountain_car_timesteps, seed, "mountain_car")
    results["mc_dqn_cpu"] = run_test("dqn", "cpu", mountain_car_timesteps, seed, "mountain_car")
    results["mc_dqn_gpu"] = run_test("dqn", "cuda", mountain_car_timesteps, seed, "mountain_car")

    # Montezuma Tests
    print("\n" + "=" * 70)
    print(" " * 25 + "Montezuma's Revenge Benchmarks")
    print("=" * 70)

    results["mr_ppo_cpu"] = run_test("ppo", "cpu", montezuma_timesteps, seed, "montezuma")
    results["mr_ppo_gpu"] = run_test("ppo", "cuda", montezuma_timesteps, seed, "montezuma")
    results["mr_dqn_cpu"] = run_test("dqn", "cpu", montezuma_timesteps, seed, "montezuma")
    results["mr_dqn_gpu"] = run_test("dqn", "cuda", montezuma_timesteps, seed, "montezuma")

    # Print summary
    print("\n" + "=" * 70)
    print(" " * 25 + "Performance Summary")
    print("=" * 70)

    # MountainCar PPO Results
    print("\n  MountainCar - PPO Results:")
    print(f"    CPU:  {format_time(results['mc_ppo_cpu']):>12s}")
    print(f"    GPU:  {format_time(results['mc_ppo_gpu']):>12s}")
    if results["mc_ppo_gpu"] > 0:
        speedup = results["mc_ppo_cpu"] / results["mc_ppo_gpu"]
        faster = "GPU" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)")

    # MountainCar DQN Results
    print("\n  MountainCar - DQN Results:")
    print(f"    CPU:  {format_time(results['mc_dqn_cpu']):>12s}")
    print(f"    GPU:  {format_time(results['mc_dqn_gpu']):>12s}")
    if results["mc_dqn_gpu"] > 0:
        speedup = results["mc_dqn_cpu"] / results["mc_dqn_gpu"]
        faster = "GPU" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)")

    # Montezuma PPO Results
    print("\n  Montezuma's Revenge - PPO Results:")
    print(f"    CPU:  {format_time(results['mr_ppo_cpu']):>12s}")
    print(f"    GPU:  {format_time(results['mr_ppo_gpu']):>12s}")
    if results["mr_ppo_gpu"] > 0:
        speedup = results["mr_ppo_cpu"] / results["mr_ppo_gpu"]
        faster = "GPU" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)")

    # Montezuma DQN Results
    print("\n  Montezuma's Revenge - DQN Results:")
    print(f"    CPU:  {format_time(results['mr_dqn_cpu']):>12s}")
    print(f"    GPU:  {format_time(results['mr_dqn_gpu']):>12s}")
    if results["mr_dqn_gpu"] > 0:
        speedup = results["mr_dqn_cpu"] / results["mr_dqn_gpu"]
        faster = "GPU" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
