#!/usr/bin/env python3
"""Benchmark script to compare GPU vs CPU performance for PPO and DQN agents."""

import sys
import time
from pathlib import Path

# Add src directory to Python path for imports (needed when running directly)
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import ale_py  # noqa: F401 - Required for ALE environment registration
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.dqn import DQNTrainer
from agents.ppo import PPOTrainer
from core.config import CONFIG
from core.seeds import reset_env_seed, set_seed


def make_mountain_car_env(shaping="none"):
    """Create and configure the Mountain Car environment."""
    env = gym.make("MountainCar-v0")
    env = RecordEpisodeStatistics(env)
    env = Monitor(env)
    return env


def make_montezuma_env(shaping="none"):
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


def run_single_benchmark(agent_type, device, timesteps, seed, env_type, shaping="none"):
    """Run a single benchmark and return the time taken."""
    env_name = "Mountain Car" if env_type == "mountain_car" else "Montezuma"
    print(f"\n{'─' * 70}", flush=True)
    print(
        f"  Testing {agent_type.upper():3s} on {device.upper():4s} ({env_name}, {shaping}, {timesteps:,} steps)",
        flush=True,
    )
    print(f"{'─' * 70}", flush=True)

    # Set seed
    actual_device = set_seed(seed, device=device, deterministic_gpu=False)
    if actual_device != device:
        print(f"  ⚠ Warning: Requested {device}, but using {actual_device}", flush=True)
        device = actual_device

    # Get config
    cfg = CONFIG.get(agent_type, {}).copy()
    cfg["total_timesteps"] = timesteps
    cfg["verbose"] = 0  # Suppress output

    # Create environment
    if env_type == "mountain_car":
        env = make_mountain_car_env(shaping)
    else:  # montezuma
        env = make_montezuma_env(shaping)
    reset_env_seed(env, seed)

    # Wrap in VecEnv
    vec_env = DummyVecEnv([lambda: env])
    if agent_type == "ppo":
        if env_type == "mountain_car":
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
            )
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

    print(f"  ✓ Completed in {format_time(elapsed_time)}", flush=True)

    return elapsed_time


def benchmark_mountain_car_ppo_none_cpu_vs_cuda(timesteps=50_000, seed=42):
    """Compare PPO Mountain Car none shaping CPU vs CUDA."""
    print("\n" + "=" * 70, flush=True)
    print(" " * 15 + "PPO MountainCar (none) - CPU vs CUDA Benchmark", flush=True)
    print("=" * 70, flush=True)

    cpu_time = run_single_benchmark(
        "ppo",
        "cpu",
        timesteps,
        seed,
        "mountain_car",
        shaping="none",
    )
    cuda_time = run_single_benchmark(
        "ppo",
        "cuda",
        timesteps,
        seed,
        "mountain_car",
        shaping="none",
    )

    print("\n  Results:", flush=True)
    print(f"    CPU:  {format_time(cpu_time):>12s}", flush=True)
    print(f"    CUDA: {format_time(cuda_time):>12s}", flush=True)
    if cuda_time > 0:
        speedup = cpu_time / cuda_time
        faster = "CUDA" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)", flush=True)

    return {"cpu": cpu_time, "cuda": cuda_time}


def test_mountain_car_ppo_none_cpu_vs_cuda(timesteps=50_000, seed=42):
    """Compare PPO Mountain Car none shaping CPU vs CUDA."""
    benchmark_mountain_car_ppo_none_cpu_vs_cuda(timesteps, seed)


def benchmark_montezuma_ppo_none_cpu_vs_cuda(timesteps=10_000, seed=42):
    """Compare PPO Montezuma's Revenge none shaping CPU vs CUDA."""
    print("\n" + "=" * 70, flush=True)
    print(
        " " * 10 + "PPO Montezuma's Revenge (none) - CPU vs CUDA Benchmark", flush=True
    )
    print("=" * 70, flush=True)

    cpu_time = run_single_benchmark(
        "ppo",
        "cpu",
        timesteps,
        seed,
        "montezuma",
        shaping="none",
    )
    cuda_time = run_single_benchmark(
        "ppo",
        "cuda",
        timesteps,
        seed,
        "montezuma",
        shaping="none",
    )

    print("\n  Results:", flush=True)
    print(f"    CPU:  {format_time(cpu_time):>12s}", flush=True)
    print(f"    CUDA: {format_time(cuda_time):>12s}", flush=True)
    if cuda_time > 0:
        speedup = cpu_time / cuda_time
        faster = "CUDA" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)", flush=True)

    return {"cpu": cpu_time, "cuda": cuda_time}


def test_montezuma_ppo_none_cpu_vs_cuda(timesteps=10_000, seed=42):
    """Compare PPO Montezuma's Revenge none shaping CPU vs CUDA."""
    benchmark_montezuma_ppo_none_cpu_vs_cuda(timesteps, seed)


def benchmark_mountain_car_dqn_none_cpu_vs_cuda(timesteps=50_000, seed=42):
    """Compare DQN MountainCar none shaping CPU vs CUDA."""
    print("\n" + "=" * 70, flush=True)
    print(" " * 15 + "DQN MountainCar (none) - CPU vs CUDA Benchmark", flush=True)
    print("=" * 70, flush=True)

    cpu_time = run_single_benchmark(
        "dqn",
        "cpu",
        timesteps,
        seed,
        "mountain_car",
        shaping="none",
    )
    cuda_time = run_single_benchmark(
        "dqn",
        "cuda",
        timesteps,
        seed,
        "mountain_car",
        shaping="none",
    )

    print("\n  Results:", flush=True)
    print(f"    CPU:  {format_time(cpu_time):>12s}", flush=True)
    print(f"    CUDA: {format_time(cuda_time):>12s}", flush=True)
    if cuda_time > 0:
        speedup = cpu_time / cuda_time
        faster = "CUDA" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)", flush=True)

    return {"cpu": cpu_time, "cuda": cuda_time}


def test_mountain_car_dqn_none_cpu_vs_cuda(timesteps=50_000, seed=42):
    """Compare DQN Mountain Car none shaping CPU vs CUDA."""
    benchmark_mountain_car_dqn_none_cpu_vs_cuda(timesteps, seed)


def benchmark_montezuma_dqn_none_cpu_vs_cuda(timesteps=10_000, seed=42):
    """Compare DQN Montezuma's Revenge none shaping CPU vs CUDA."""
    print("\n" + "=" * 70, flush=True)
    print(
        " " * 10 + "DQN Montezuma's Revenge (none) - CPU vs CUDA Benchmark", flush=True
    )
    print("=" * 70, flush=True)

    cpu_time = run_single_benchmark(
        "dqn",
        "cpu",
        timesteps,
        seed,
        "montezuma",
        shaping="none",
    )
    cuda_time = run_single_benchmark(
        "dqn",
        "cuda",
        timesteps,
        seed,
        "montezuma",
        shaping="none",
    )

    print("\n  Results:", flush=True)
    print(f"    CPU:  {format_time(cpu_time):>12s}", flush=True)
    print(f"    CUDA: {format_time(cuda_time):>12s}", flush=True)
    if cuda_time > 0:
        speedup = cpu_time / cuda_time
        faster = "CUDA" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)", flush=True)

    return {"cpu": cpu_time, "cuda": cuda_time}


def test_montezuma_dqn_none_cpu_vs_cuda(timesteps=10_000, seed=42):
    """Compare DQN Montezuma's Revenge none shaping CPU vs CUDA."""
    benchmark_montezuma_dqn_none_cpu_vs_cuda(timesteps, seed)


def main():
    """Run all performance benchmarks."""
    mountain_car_timesteps = 50_000
    montezuma_timesteps = 10_000
    seed = 42

    print("\n" + "=" * 70, flush=True)
    print(" " * 20 + "GPU vs CPU Performance Benchmark", flush=True)
    print("=" * 70, flush=True)
    print(f"  MountainCar timesteps: {mountain_car_timesteps:,}", flush=True)
    print(f"  Montezuma timesteps: {montezuma_timesteps:,}", flush=True)
    print(f"  Seed: {seed}", flush=True)
    print("  Shaping: none (baseline)", flush=True)
    print("=" * 70, flush=True)

    # Run all benchmarks and collect results
    results = {}
    results["mc_ppo"] = benchmark_mountain_car_ppo_none_cpu_vs_cuda(
        mountain_car_timesteps, seed
    )
    results["mr_ppo"] = benchmark_montezuma_ppo_none_cpu_vs_cuda(
        montezuma_timesteps, seed
    )
    results["mc_dqn"] = benchmark_mountain_car_dqn_none_cpu_vs_cuda(
        mountain_car_timesteps, seed
    )
    results["mr_dqn"] = benchmark_montezuma_dqn_none_cpu_vs_cuda(
        montezuma_timesteps, seed
    )

    # Print final summary
    print("\n" + "=" * 70, flush=True)
    print(" " * 25 + "Final Benchmark Summary", flush=True)
    print("=" * 70, flush=True)

    # MountainCar PPO Results
    print("\n  Mountain Car - PPO Results:", flush=True)
    print(f"    CPU:  {format_time(results['mc_ppo']['cpu']):>12s}", flush=True)
    print(f"    CUDA: {format_time(results['mc_ppo']['cuda']):>12s}", flush=True)
    if results["mc_ppo"]["cuda"] > 0:
        speedup = results["mc_ppo"]["cpu"] / results["mc_ppo"]["cuda"]
        faster = "CUDA" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)", flush=True)

    # MountainCar DQN Results
    print("\n  Mountain Car - DQN Results:", flush=True)
    print(f"    CPU:  {format_time(results['mc_dqn']['cpu']):>12s}", flush=True)
    print(f"    CUDA: {format_time(results['mc_dqn']['cuda']):>12s}", flush=True)
    if results["mc_dqn"]["cuda"] > 0:
        speedup = results["mc_dqn"]["cpu"] / results["mc_dqn"]["cuda"]
        faster = "CUDA" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)", flush=True)

    # Montezuma PPO Results
    print("\n  Montezuma's Revenge - PPO Results:", flush=True)
    print(f"    CPU:  {format_time(results['mr_ppo']['cpu']):>12s}", flush=True)
    print(f"    CUDA: {format_time(results['mr_ppo']['cuda']):>12s}", flush=True)
    if results["mr_ppo"]["cuda"] > 0:
        speedup = results["mr_ppo"]["cpu"] / results["mr_ppo"]["cuda"]
        faster = "CUDA" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)", flush=True)

    # Montezuma DQN Results
    print("\n  Montezuma's Revenge - DQN Results:", flush=True)
    print(f"    CPU:  {format_time(results['mr_dqn']['cpu']):>12s}", flush=True)
    print(f"    CUDA: {format_time(results['mr_dqn']['cuda']):>12s}", flush=True)
    if results["mr_dqn"]["cuda"] > 0:
        speedup = results["mr_dqn"]["cpu"] / results["mr_dqn"]["cuda"]
        faster = "CUDA" if speedup > 1 else "CPU"
        print(f"    Speedup: {speedup:.2f}x ({faster} faster)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(" " * 25 + "All Benchmarks Completed", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
