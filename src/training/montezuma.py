#!/usr/bin/env python3
"""Training script for Montezuma's Revenge with all configuration options."""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import ale_py  # noqa: F401
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from core.config import CONFIG
from core.seeds import reset_env_seed, set_seed
from core.wandb_callback import WandbCallback
from environments.montezuma import make_montezuma
from util import PROJECT_ROOT_PATH


def train_ppo_montezuma(
    device,
    timesteps,
    seed=42,
    run=None,
    observation_type="default",
    include_noop=True,
    checkpoint_path=None,
    checkpoint_interval=100_000,
    enable_distance_reward=False,
    distance_reward_scale=0.1,
):
    """
    Train PPO on Montezuma's Revenge with unified environment.

    Args:
        device: Device to use
        timesteps: Number of training timesteps
        seed: Random seed
        run: wandb run object
        observation_type: Type of observations (default, grey, ram, ram-only, grey-ram)
        include_noop: If True, include NOOP action (7 actions), else 6 actions
        checkpoint_path: Path to checkpoint directory for saving models during training
        checkpoint_interval: Save checkpoint every N timesteps
        enable_distance_reward: If True, add reward shaping
        distance_reward_scale: Scale factor for distance reward

    Returns:
        Tuple of (model, vec_env) for saving VecNormalize stats
    """
    print(f"\n{'=' * 70}")
    print("Training PPO on Montezuma's Revenge")
    print(f"Device: {device.upper()}, Timesteps: {timesteps:,}")
    print(f"Observation type: {observation_type}")
    print(f"Action space: {'7 actions (with NOOP)' if include_noop else '6 actions (no NOOP)'}")
    print(f"{'=' * 70}")

    # Set seed
    actual_device = set_seed(seed, device=device, deterministic_gpu=False)
    if actual_device != device:
        print(f"  ⚠ Warning: Requested {device}, but using {actual_device}")
        device = actual_device

    # Get config
    cfg = CONFIG.get("ppo", {}).copy()
    cfg["total_timesteps"] = timesteps
    cfg["verbose"] = 1

    # Create environment
    env = make_montezuma(
        render_mode="rgb_array",
        observation_type=observation_type,
        include_noop=include_noop,
        enable_distance_reward=enable_distance_reward,
        distance_reward_scale=distance_reward_scale,
    )
    reset_env_seed(env, seed)

    # Wrap for SB3
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    # Add video recording if wandb run is available
    if run is not None:
        video_recording_interval = 10000
        video_length = 600
        video_folder = os.path.join("videos", "montezuma", run.id)
        os.makedirs(video_folder, exist_ok=True)
        vec_env = VecVideoRecorder(
            vec_env,
            video_folder,
            record_video_trigger=lambda x: x % video_recording_interval == 0,
            video_length=video_length,
        )

    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
    )

    # Determine policy type based on observation type
    if observation_type == "ram":
        # Dictionary observation requires MultiInputPolicy
        policy_type = "MultiInputPolicy"
    else:
        # All other types are flattened, use MlpPolicy
        policy_type = "MlpPolicy"

    # Create PPO model
    model = PPO(
        policy_type,
        vec_env,
        device=device,
        learning_rate=cfg.get("learning_rate", 3e-4),
        n_steps=cfg.get("n_steps", 2048),
        batch_size=cfg.get("batch_size", 64),
        gamma=cfg.get("gamma", 0.99),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        verbose=cfg.get("verbose", 1),
        seed=seed,
    )

    # Setup callbacks
    callbacks = []
    if run:
        callbacks.append(WandbCallback(run))

    # Add checkpoint callback (always enabled if checkpoint_path is provided)
    if checkpoint_path:
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=checkpoint_path,
            name_prefix="checkpoint",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

    callback = CallbackList(callbacks) if callbacks else None

    # Train
    model.learn(
        total_timesteps=cfg.get("total_timesteps", timesteps),
        callback=callback,
    )

    # Manually save VecNormalize stats for all checkpoints (if CheckpointCallback didn't)
    if checkpoint_path:
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "checkpoint_*.zip"))
        for checkpoint_file in checkpoint_files:
            vecnormalize_path = checkpoint_file.replace(".zip", "_vecnormalize.pkl")
            if not os.path.exists(vecnormalize_path):
                vec_env.save(vecnormalize_path)

    print(f"  ✓ Completed: Montezuma on {device}\n")
    return model, vec_env


def load_and_continue_training(
    checkpoint_path,
    device,
    additional_timesteps=1_000_000,
    seed=42,
    run=None,
    observation_type="default",
    include_noop=True,
    checkpoint_interval=100_000,
    save_checkpoint_path=None,
    enable_distance_reward=False,
    distance_reward_scale=0.1,
):
    """
    Load a checkpoint and continue training.

    Args:
        checkpoint_path: Path to the checkpoint file (.zip) to load
        device: Device to use
        additional_timesteps: Number of additional training timesteps
        seed: Random seed (should match original training)
        run: wandb run object
        observation_type: Type of observations (must match original training)
        include_noop: If True, include NOOP action (must match original training)
        checkpoint_interval: Save checkpoint every N timesteps
        save_checkpoint_path: Path to checkpoint directory for saving models during training
        enable_distance_reward: If True, add reward shaping
        distance_reward_scale: Scale factor for distance reward

    Returns:
        Tuple of (model, vec_env)
    """
    print(f"\n{'=' * 70}")
    print("Loading checkpoint and continuing training")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device.upper()}, Training timesteps: {additional_timesteps:,}")
    print(f"{'=' * 70}")

    # Set seed (should match original training)
    actual_device = set_seed(seed, device=device, deterministic_gpu=False)
    if actual_device != device:
        print(f"  ⚠ Warning: Requested {device}, but using {actual_device}")
        device = actual_device

    # Create environment (must match original training setup)
    env = make_montezuma(
        render_mode="rgb_array",
        observation_type=observation_type,
        include_noop=include_noop,
        enable_distance_reward=enable_distance_reward,
        distance_reward_scale=distance_reward_scale,
    )
    reset_env_seed(env, seed)

    # Wrap for SB3
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    # Add video recording if wandb run is available
    if run is not None:
        video_recording_interval = 10000
        video_length = 600
        video_folder = os.path.join("videos", "montezuma", run.id)
        os.makedirs(video_folder, exist_ok=True)
        vec_env = VecVideoRecorder(
            vec_env,
            video_folder,
            record_video_trigger=lambda x: x % video_recording_interval == 0,
            video_length=video_length,
        )

    # Normalize observations and rewards (must match original setup)
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
    )

    # Load VecNormalize stats first if they exist (before loading model)
    vecnormalize_path = checkpoint_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from {vecnormalize_path}...")
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        print("✓ VecNormalize stats loaded")
    else:
        print("⚠ VecNormalize stats not found, will use new normalization")

    # Load model from checkpoint (with correct environment)
    print(f"Loading model from {checkpoint_path}...")
    model = PPO.load(checkpoint_path, env=vec_env, device=device)

    # Get timesteps from checkpoint
    checkpoint_timesteps = model.num_timesteps
    total_timesteps_after = checkpoint_timesteps + additional_timesteps

    print("✓ Model loaded successfully")
    print(
        f"\nTotal Timesteps trained: {total_timesteps_after:,} ({checkpoint_timesteps:,} from checkpoint + {additional_timesteps:,} training_timesteps)"
    )
    print(f"{'=' * 70}\n")

    # Setup callbacks
    callbacks = []
    if run:
        callbacks.append(WandbCallback(run))

    # Add checkpoint callback if save_checkpoint_path is provided
    if save_checkpoint_path:
        os.makedirs(save_checkpoint_path, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=save_checkpoint_path,
            name_prefix="checkpoint",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

    callback = CallbackList(callbacks) if callbacks else None

    # Continue training
    print(f"Continuing training for {additional_timesteps:,} training_timesteps...")
    model.learn(
        total_timesteps=additional_timesteps,
        callback=callback,
        reset_num_timesteps=False,  # Important: don't reset timestep counter
    )

    # Manually save VecNormalize stats for all checkpoints created during continued training
    if save_checkpoint_path:
        checkpoint_files = glob.glob(
            os.path.join(save_checkpoint_path, "checkpoint_*.zip")
        )
        for checkpoint_file in checkpoint_files:
            vecnormalize_path = checkpoint_file.replace(".zip", "_vecnormalize.pkl")
            if not os.path.exists(vecnormalize_path):
                vec_env.save(vecnormalize_path)

    final_timesteps = model.num_timesteps
    print(f"  ✓ Completed training on {device}")
    print(f"  Total Timesteps trained: {final_timesteps:,}\n")
    return model, vec_env


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on Montezuma's Revenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    # Environment configuration
    parser.add_argument(
        "--observation-type",
        type=str,
        default="default",
        choices=["default", "grey", "ram", "ram-only", "grey-ram"],
        help="Type of observations",
    )
    parser.add_argument(
        "--include-noop",
        action="store_true",
        default=True,
        help="Include NOOP action (7 actions). Use --no-include-noop for 6 actions",
    )
    parser.add_argument(
        "--no-include-noop",
        dest="include_noop",
        action="store_false",
        help="Exclude NOOP action (6 actions for speedrunning)",
    )

    # Reward shaping
    parser.add_argument(
        "--enable-distance-reward",
        action="store_true",
        help="Enable distance-based reward shaping",
    )
    parser.add_argument(
        "--distance-reward-scale",
        type=float,
        default=0.1,
        help="Scale factor for distance reward",
    )

    # Checkpointing (always enabled)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100_000,
        help="Save checkpoint every N timesteps",
    )

    # Continue training
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Path to checkpoint file to continue training from",
    )
    parser.add_argument(
        "--save-checkpoints",
        type=str,
        default=None,
        help="Path to checkpoint directory for saving during continued training",
    )

    # Wandb
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    args = parser.parse_args()

    # Ensure GPU is available if requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Initialize wandb run
    run = None
    if not args.no_wandb:
        run_name = f"montezuma_{args.observation_type}_{args.timesteps}steps"
        if not args.include_noop:
            run_name += "_no-noop"
        run = wandb.init(
            project="Montezuma",
            name=run_name,
            config={
                "observation_type": args.observation_type,
                "include_noop": args.include_noop,
                "action_space_size": 7 if args.include_noop else 6,
                "timesteps": args.timesteps,
                "seed": args.seed,
                "device": args.device,
                "enable_distance_reward": args.enable_distance_reward,
                "distance_reward_scale": args.distance_reward_scale,
                "checkpoint_interval": args.checkpoint_interval,
                **CONFIG.get("ppo", {}),
            },
            reinit=True,
        )

    # Continue training or start new training
    if args.continue_from:
        model, vec_env = load_and_continue_training(
            checkpoint_path=args.continue_from,
            device=args.device,
            additional_timesteps=args.timesteps,
            seed=args.seed,
            run=run,
            observation_type=args.observation_type,
            include_noop=args.include_noop,
            checkpoint_interval=args.checkpoint_interval,
            save_checkpoint_path=args.save_checkpoints or args.checkpoint_path,
            enable_distance_reward=args.enable_distance_reward,
            distance_reward_scale=args.distance_reward_scale,
        )
    else:
        model, vec_env = train_ppo_montezuma(
            device=args.device,
            timesteps=args.timesteps,
            seed=args.seed,
            run=run,
            observation_type=args.observation_type,
            include_noop=args.include_noop,
            checkpoint_path=args.checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            enable_distance_reward=args.enable_distance_reward,
            distance_reward_scale=args.distance_reward_scale,
        )

    # Save final model
    save_dir = os.environ.get("SHAPINGRL_SAVE_DIR", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    model_name = f"montezuma_{args.observation_type}_{args.timesteps}steps_seed{args.seed}"
    if not args.include_noop:
        model_name += "_no-noop"
    model_path = os.path.join(save_dir, f"{model_name}.zip")
    model.save(model_path)
    print(f"✓ Final model saved to {model_path}")

    # Save VecNormalize stats
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    vec_env.save(vecnormalize_path)
    print(f"✓ VecNormalize stats saved to {vecnormalize_path}")

    # Upload videos to wandb if available
    if run:
        def extract_steps(filename):
            match = re.search(r"step-(\d+)-to-step-(\d+)", filename)
            if match:
                return tuple(map(int, match.groups()))
            return None

        video_files = sorted(
            glob.glob(
                os.path.join(
                    PROJECT_ROOT_PATH, "videos", "montezuma", run.id, "*.mp4"
                )
            )
        )

        for path in video_files:
            step_range = extract_steps(path)
            if step_range:
                start_step, end_step = step_range
                wandb.log(
                    {
                        "gym_video": wandb.Video(
                            path, caption="Video", fps=30, format="mp4"
                        ),
                        "gym_video_start_step": start_step,
                        "gym_video_end_step": end_step,
                        "env_step": start_step,
                    }
                )
        run.finish()

    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()

