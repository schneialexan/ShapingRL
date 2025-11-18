#!/usr/bin/env python3
"""Montezuma's Revenge environment with grayscale state space and training."""

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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO on Montezuma's Revenge (grayscale state space)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--timesteps", type=int, default=1_000_000, help="Number of training timesteps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--shaping",
        type=str,
        default=None,
        choices=[None, "manhattan"],
        help="Reward shaping method",
    )
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
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--project", default="Montezuma Grey", help="WandB project name"
    )

    return parser.parse_args()


def train_ppo_montezuma_grey(
    device,
    timesteps,
    seed=42,
    run=None,
    checkpoint_path=None,
    checkpoint_interval=100_000,
    shaping=None,
):
    """Train PPO on Montezuma's Revenge with grayscale state space."""
    print(f"\n{'=' * 70}")
    print("Training PPO on Montezuma's Revenge (Grayscale)")
    print(f"Device: {device.upper()}, Timesteps: {timesteps:,}")
    print("Action space: 7 actions (with NOOP)")
    if shaping:
        print(f"Reward shaping: {shaping}")
    print(f"{'=' * 70}")

    actual_device = set_seed(seed, device=device, deterministic_gpu=False)
    if actual_device != device:
        print(f"  ⚠ Warning: Requested {device}, but using {actual_device}")
        device = actual_device

    cfg = CONFIG.get("ppo", {}).copy()
    cfg["total_timesteps"] = timesteps
    cfg["verbose"] = 1

    # Create environment with fixed grayscale state space (NOOP always included)
    env = make_montezuma(
        render_mode="rgb_array",
        state_space="grey",
        shaping=shaping,
    )
    reset_env_seed(env, seed)

    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

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

    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
    )

    # Grayscale uses MlpPolicy (flattened observations)
    policy_type = "MlpPolicy"

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

    callbacks = []
    if run:
        callbacks.append(WandbCallback(run))

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

    model.learn(
        total_timesteps=cfg.get("total_timesteps", timesteps),
        callback=callback,
    )

    if checkpoint_path:
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "checkpoint_*.zip"))
        for checkpoint_file in checkpoint_files:
            vecnormalize_path = checkpoint_file.replace(".zip", "_vecnormalize.pkl")
            if not os.path.exists(vecnormalize_path):
                vec_env.save(vecnormalize_path)

    print(f"  ✓ Completed: Montezuma Grey on {device}\n")
    return model, vec_env


def load_and_continue_training_grey(
    checkpoint_path,
    device,
    additional_timesteps=1_000_000,
    seed=42,
    run=None,
    checkpoint_interval=100_000,
    save_checkpoint_path=None,
    shaping=None,
):
    """Load a checkpoint and continue training (grayscale)."""
    print(f"\n{'=' * 70}")
    print("Loading checkpoint and continuing training (Grayscale)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device.upper()}, Training timesteps: {additional_timesteps:,}")
    print(f"{'=' * 70}")

    actual_device = set_seed(seed, device=device, deterministic_gpu=False)
    if actual_device != device:
        print(f"  ⚠ Warning: Requested {device}, but using {actual_device}")
        device = actual_device

    env = make_montezuma(
        render_mode="rgb_array",
        state_space="grey",
        include_noop=True,  # Always include NOOP
        shaping=shaping,
    )
    reset_env_seed(env, seed)

    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

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

    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
    )

    vecnormalize_path = checkpoint_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from {vecnormalize_path}...")
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        print("✓ VecNormalize stats loaded")
    else:
        print("⚠ VecNormalize stats not found, will use new normalization")

    print(f"Loading model from {checkpoint_path}...")
    model = PPO.load(checkpoint_path, env=vec_env, device=device)

    checkpoint_timesteps = model.num_timesteps
    total_timesteps_after = checkpoint_timesteps + additional_timesteps

    print("✓ Model loaded successfully")
    print(
        f"\nTotal Timesteps trained: {total_timesteps_after:,} ({checkpoint_timesteps:,} from checkpoint + {additional_timesteps:,} training_timesteps)"
    )
    print(f"{'=' * 70}\n")

    callbacks = []
    if run:
        callbacks.append(WandbCallback(run))

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

    print(f"Continuing training for {additional_timesteps:,} training_timesteps...")
    model.learn(
        total_timesteps=additional_timesteps,
        callback=callback,
        reset_num_timesteps=False,
    )

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
    """Main training function."""
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    run = None
    if not args.no_wandb:
        run_name = f"grey_{args.timesteps}steps"
        if args.shaping:
            run_name += f"_{args.shaping}"
        run = wandb.init(
            project=args.project,
            name=run_name,
            config={
                "state_space": "grey",
                "action_space_size": 7,  # Always with NOOP
                "timesteps": args.timesteps,
                "seed": args.seed,
                "device": args.device,
                "shaping": args.shaping,
                "checkpoint_interval": args.checkpoint_interval,
                **CONFIG.get("ppo", {}),
            },
            reinit=True,
        )

    if args.continue_from:
        model, vec_env = load_and_continue_training_grey(
            checkpoint_path=args.continue_from,
            device=args.device,
            additional_timesteps=args.timesteps,
            seed=args.seed,
            run=run,
            checkpoint_interval=args.checkpoint_interval,
            save_checkpoint_path=args.save_checkpoints or args.checkpoint_path,
            shaping=args.shaping,
        )
    else:
        model, vec_env = train_ppo_montezuma_grey(
            device=args.device,
            timesteps=args.timesteps,
            seed=args.seed,
            run=run,
            checkpoint_path=args.checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            shaping=args.shaping,
        )

    save_dir = os.environ.get("SHAPINGRL_SAVE_DIR", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    model_name = f"montezuma_grey_{args.timesteps}steps_seed{args.seed}"
    if args.shaping:
        model_name += f"_{args.shaping}"
    model_path = os.path.join(save_dir, f"{model_name}.zip")
    model.save(model_path)
    print(f"✓ Final model saved to {model_path}")

    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    vec_env.save(vecnormalize_path)
    print(f"✓ VecNormalize stats saved to {vecnormalize_path}")

    if run:

        def extract_steps(filename):
            match = re.search(r"step-(\d+)-to-step-(\d+)", filename)
            if match:
                return tuple(map(int, match.groups()))
            return None

        video_files = sorted(
            glob.glob(
                os.path.join(PROJECT_ROOT_PATH, "videos", "montezuma", run.id, "*.mp4")
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
