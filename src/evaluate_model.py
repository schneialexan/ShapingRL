#!/usr/bin/env python3
"""Evaluate a trained model and create a final evaluation video."""

import argparse
import os

import ale_py  # noqa: F401 - Required for ALE environment registration
import gymnasium as gym
import wandb
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecTransposeImage,
    VecVideoRecorder,
)

from core.seeds import reset_env_seed, set_seed


def make_env(render_mode="rgb_array"):
    """Create Montezuma's Revenge environment without shaping."""
    env_name = "ALE/MontezumaRevenge-v5"
    # Check if environment is already registered
    if env_name not in [spec.id for spec in gym.envs.registry.values()]:
        register(
            id=env_name,
            entry_point="ale_py.roms:MontezumaRevenge",
            max_episode_steps=4_000,
            reward_threshold=3000.0,
        )
    env = gym.make(env_name, render_mode=render_mode)
    return env


def evaluate_model(
    model_path,
    policy_type="MlpPolicy",
    num_episodes=5,
    max_steps_per_episode=4000,
    seed=42,
    device="cpu",
    output_dir="evaluation_videos",
    upload_to_wandb=False,
    wandb_run_id=None,
):
    """
    Evaluate a trained model and create evaluation videos.

    Args:
        model_path: Path to the trained model file
        policy_type: "MlpPolicy" or "CnnPolicy" (used to determine environment setup)
        num_episodes: Number of episodes to evaluate
        max_steps_per_episode: Maximum steps per episode
        seed: Random seed
        device: Device to use ("cpu" or "cuda")
        output_dir: Directory to save evaluation videos
        upload_to_wandb: Whether to upload videos to wandb
        wandb_run_id: wandb run ID to associate videos with (optional)
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating model: {model_path}")
    print(f"Policy type: {policy_type}, Episodes: {num_episodes}, Seed: {seed}")
    print(f"{'=' * 70}\n")

    # Set seed
    set_seed(seed, device=device, deterministic_gpu=False)

    # Load model
    print(f"Loading model from {model_path}...")
    if model_path.endswith(".zip"):
        # PPO or DQN model
        if "ppo" in model_path.lower() or "PPO" in model_path:
            model = PPO.load(model_path, device=device)
        elif "dqn" in model_path.lower() or "DQN" in model_path:
            model = DQN.load(model_path, device=device)
        else:
            # Try PPO first, fallback to DQN
            try:
                model = PPO.load(model_path, device=device)
            except Exception:
                model = DQN.load(model_path, device=device)
    else:
        raise ValueError(f"Unknown model format: {model_path}")

    print("✓ Model loaded successfully\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb if requested
    run = None
    if upload_to_wandb:
        run = wandb.init(
            project="Montezuma's Revenge",
            name=f"evaluation_{os.path.basename(model_path)}",
            job_type="evaluation",
            reinit=True,
        )
        if wandb_run_id:
            run.config.update({"original_run_id": wandb_run_id})

    # Evaluate episodes
    total_reward = 0
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Create environment for this episode
        env = make_env(render_mode="rgb_array")
        reset_env_seed(env, seed + episode)

        # Wrap environment
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])

        # Setup environment based on policy type
        # Note: VecNormalize stats are saved with the model, but for evaluation
        # we use fresh normalization (or load from model if available)
        if policy_type == "CnnPolicy":
            vec_env = VecTransposeImage(vec_env)
            vec_env = VecNormalize(
                vec_env,
                norm_obs=False,  # CNN handles normalization internally
                norm_reward=True,
                clip_reward=500,
            )
        else:
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
            )

        # Try to load normalization stats from model if available
        # (This is optional - model should work without it, but might perform slightly differently)
        try:
            if hasattr(model, "env") and hasattr(model.env, "norm_reward"):
                # Model was trained with VecNormalize, try to sync stats
                # Note: This is a simplified approach - full sync would require loading vec_normalize.pkl
                pass
        except Exception:
            pass  # Continue without syncing stats

        # Create video recorder for this episode
        # VecVideoRecorder records when trigger returns True
        # We want to record the entire episode, so trigger at step 0
        vec_env = VecVideoRecorder(
            vec_env,
            output_dir,
            record_video_trigger=lambda x: x == 0,  # Record from start
            video_length=max_steps_per_episode,
            name_prefix=f"episode_{episode + 1}",
        )

        # Run episode
        obs, info = vec_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vec_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        vec_env.close()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_reward += episode_reward

        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")

        # Find and upload video to wandb
        if run:
            # VecVideoRecorder creates files with pattern: {name_prefix}-{step_range}.mp4
            video_pattern = os.path.join(output_dir, f"episode_{episode + 1}-*.mp4")
            import glob

            video_files = glob.glob(video_pattern)
            if video_files:
                video_path = video_files[0]  # Take first match
                wandb.log(
                    {
                        "evaluation_video": wandb.Video(
                            video_path,
                            caption=f"Evaluation Episode {episode + 1}",
                            fps=30,
                            format="mp4",
                        ),
                        "episode": episode + 1,
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                    }
                )

    # Print summary
    avg_reward = total_reward / num_episodes
    avg_length = sum(episode_lengths) / len(episode_lengths)

    print(f"\n{'=' * 70}")
    print("Evaluation Summary:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Episode Length: {avg_length:.1f}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Videos saved to: {output_dir}")
    print(f"{'=' * 70}\n")

    if run:
        wandb.log(
            {
                "evaluation_avg_reward": avg_reward,
                "evaluation_avg_length": avg_length,
                "evaluation_total_reward": total_reward,
            }
        )
        run.finish()

    return {
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "total_reward": total_reward,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model and create evaluation videos"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model file (.zip)",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        choices=["MlpPolicy", "CnnPolicy"],
        default="MlpPolicy",
        help="Policy type used in the model (MlpPolicy or CnnPolicy)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_videos",
        help="Directory to save evaluation videos",
    )
    parser.add_argument(
        "--upload-wandb",
        action="store_true",
        help="Upload videos to wandb",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="wandb run ID to associate evaluation with",
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        policy_type=args.policy_type,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        upload_to_wandb=args.upload_wandb,
        wandb_run_id=args.wandb_run_id,
    )


if __name__ == "__main__":
    main()
