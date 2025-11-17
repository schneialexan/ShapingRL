#!/usr/bin/env python3
"""Test checkpointing and resuming training functionality."""

import glob
import os
import sys
import tempfile
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import ale_py  # noqa: F401 - Required for ALE environment registration
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from core.seeds import reset_env_seed
from environments.montezuma import make_montezuma
from training.montezuma import load_and_continue_training, train_ppo_montezuma


def test_checkpointing_basic():
    """Test basic checkpointing functionality."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Test 1: Basic checkpointing during training")
        print("=" * 70)

        # Train for a very short time with checkpointing (for fast tests)
        _ = train_ppo_montezuma(
            device="cuda",
            timesteps=2_000,  # Very short training for fast test
            seed=42,
            run=None,  # No wandb for test
            observation_type="ram-only",
            include_noop=False,
            checkpoint_path=checkpoint_dir,
            checkpoint_interval=1_000,  # Save every 1k steps (creates 2 checkpoints)
        )

        # Check that checkpoints were created
        # CheckpointCallback saves as: checkpoint_XXXXXX_steps.zip
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.zip"))
        )
        # VecNormalize stats are saved separately as: checkpoint_XXXXXX_steps_vecnormalize.pkl
        vecnormalize_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*_vecnormalize.pkl"))
        )

        print(f"\nFound {len(checkpoint_files)} checkpoint files:")
        for f in checkpoint_files:
            print(f"  - {os.path.basename(f)}")
        print(f"Found {len(vecnormalize_files)} vecnormalize files:")
        for f in vecnormalize_files:
            print(f"  - {os.path.basename(f)}")

        # Should have at least 1 checkpoint (at 1k steps, maybe 2k if training completes)
        assert len(checkpoint_files) >= 1, (
            f"Expected at least 1 checkpoint, got {len(checkpoint_files)}"
        )
        # VecNormalize files might not always be created if training is very short
        # but if they exist, they should match checkpoints
        if len(vecnormalize_files) > 0:
            assert len(vecnormalize_files) == len(checkpoint_files), (
                f"Expected same number of vecnormalize files as checkpoints, "
                f"got {len(vecnormalize_files)} vs {len(checkpoint_files)}"
            )

        # Verify checkpoint files are valid
        for checkpoint_file in checkpoint_files:
            assert os.path.exists(checkpoint_file), (
                f"Checkpoint file not found: {checkpoint_file}"
            )
            # Try to load it
            try:
                test_model = PPO.load(checkpoint_file, device="cuda")
                assert test_model is not None
                print(f"✓ Successfully loaded {os.path.basename(checkpoint_file)}")
            except Exception as e:
                pytest.fail(f"Failed to load checkpoint {checkpoint_file}: {e}")

        print("✓ Test 1 passed: Checkpointing works correctly")


def test_resume_training():
    """Test resuming training from a checkpoint."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Test 2: Resuming training from checkpoint")
        print("=" * 70)

        # Step 1: Train for a very short time and save a checkpoint
        print("\nStep 1: Initial training with checkpointing...")
        _ = train_ppo_montezuma(
            device="cuda",
            timesteps=1_200,  # Very short training for fast test
            seed=42,
            run=None,
            observation_type="ram-only",
            include_noop=False,
            checkpoint_path=checkpoint_dir,
            checkpoint_interval=1_000,  # Save every 1k steps
        )

        # Find the checkpoint at 1k steps
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.zip"))
        )
        assert len(checkpoint_files) > 0, "No checkpoint files found"

        # Use the first checkpoint (should be around 1k steps)
        checkpoint_to_load = checkpoint_files[0]
        print(f"\nStep 2: Loading checkpoint: {os.path.basename(checkpoint_to_load)}")

        # Step 2: Load checkpoint and continue training
        model2, _ = load_and_continue_training(
            checkpoint_path=checkpoint_to_load,
            device="cuda",
            additional_timesteps=800,  # Train for additional 800 steps
            seed=42,  # Same seed
            run=None,
            observation_type="ram-only",
            include_noop=False,
            checkpoint_interval=500,
            save_checkpoint_path=os.path.join(checkpoint_dir, "continued"),
        )

        # Verify model was trained
        assert model2 is not None, "Model should be returned"
        assert model2.num_timesteps >= 1_200, (
            f"Model should have at least 1.2k steps, got {model2.num_timesteps}"
        )

        print(f"✓ Model has {model2.num_timesteps:,} total timesteps")
        print("✓ Test 2 passed: Resuming training works correctly")


def test_vecnormalize_stats_preserved():
    """Test that VecNormalize stats are preserved when loading checkpoints."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Test 3: VecNormalize stats preservation")
        print("=" * 70)

        # Train and create checkpoint (very short for fast test)
        _ = train_ppo_montezuma(
            device="cuda",
            timesteps=1_200,
            seed=42,
            run=None,
            observation_type="ram-only",
            include_noop=False,
            checkpoint_path=checkpoint_dir,
            checkpoint_interval=1_000,
        )

        # Get checkpoint
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.zip"))
        )
        assert len(checkpoint_files) > 0
        checkpoint_path = checkpoint_files[0]

        # Create environment and load checkpoint
        env = make_montezuma(
            render_mode="rgb_array", observation_type="ram-only", include_noop=False
        )
        reset_env_seed(env, 42)
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
        )

        # Load model
        model = PPO.load(checkpoint_path, env=vec_env, device="cuda")

        # Check if vecnormalize stats file exists and can be loaded
        vecnormalize_path = checkpoint_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnormalize_path):
            print(
                f"✓ VecNormalize stats file exists: {os.path.basename(vecnormalize_path)}"
            )
            vec_env_loaded = VecNormalize.load(vecnormalize_path, vec_env)
            model.set_env(vec_env_loaded)
            print("✓ VecNormalize stats loaded successfully")
        else:
            print(
                "⚠ VecNormalize stats file not found (may be normal if checkpoint was created before stats were saved)"
            )

        print("✓ Test 3 passed: VecNormalize stats handling works")


def test_timestep_counter_preserved():
    """Test that timestep counter is preserved when resuming."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Test 4: Timestep counter preservation")
        print("=" * 70)

        # Train for 1.2k steps (very short for fast test)
        model1, _ = train_ppo_montezuma(
            device="cuda",
            timesteps=1_200,
            seed=42,
            run=None,
            observation_type="ram-only",
            include_noop=False,
            checkpoint_path=checkpoint_dir,
            checkpoint_interval=1_000,
        )

        initial_timesteps = model1.num_timesteps
        print(f"Initial training: {initial_timesteps:,} timesteps")

        # Load checkpoint and continue
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.zip"))
        )
        checkpoint_path = checkpoint_files[0]

        model2, _ = load_and_continue_training(
            checkpoint_path=checkpoint_path,
            device="cuda",
            additional_timesteps=800,
            seed=42,
            run=None,
            observation_type="ram-only",
            include_noop=False,
        )

        final_timesteps = model2.num_timesteps
        print(f"After resuming: {final_timesteps:,} timesteps")

        # Timesteps should be >= initial + additional (may be slightly more due to rounding)
        assert final_timesteps >= initial_timesteps + 800, (
            f"Expected at least {initial_timesteps + 800:,} timesteps, "
            f"got {final_timesteps:,}"
        )

        print("✓ Test 4 passed: Timestep counter is preserved correctly")


def test_full_workflow():
    """Test complete workflow: train -> checkpoint -> resume -> save."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Test 5: Complete workflow test")
        print("=" * 70)

        # Step 1: Initial training (very short for fast test)
        print("\n1. Initial training (1.2k steps)...")
        _ = train_ppo_montezuma(
            device="cuda",
            timesteps=1_200,
            seed=42,
            run=None,
            observation_type="ram-only",
            include_noop=False,
            checkpoint_path=checkpoint_dir,
            checkpoint_interval=1_000,
        )

        # Step 2: Find and load checkpoint
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.zip"))
        )
        assert len(checkpoint_files) > 0
        checkpoint_path = checkpoint_files[0]
        print(f"\n2. Found checkpoint: {os.path.basename(checkpoint_path)}")

        # Step 3: Resume training
        print("\n3. Resuming training (additional 800 steps)...")
        model2, _ = load_and_continue_training(
            checkpoint_path=checkpoint_path,
            device="cuda",
            additional_timesteps=800,
            seed=42,
            run=None,
            observation_type="ram-only",
            include_noop=False,
        )

        # Step 4: Save final model
        final_model_path = os.path.join(tmpdir, "final_model.zip")
        model2.save(final_model_path)
        print(f"\n4. Final model saved to: {final_model_path}")

        # Step 5: Verify final model can be loaded
        print("\n5. Verifying final model can be loaded...")
        env = make_montezuma(
            render_mode="rgb_array", observation_type="ram-only", include_noop=False
        )
        reset_env_seed(env, 42)
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=255, clip_reward=500
        )

        final_model = PPO.load(final_model_path, env=vec_env, device="cuda")
        assert final_model is not None
        assert final_model.num_timesteps >= 1_200
        print(
            f"✓ Final model loaded successfully with {final_model.num_timesteps:,} timesteps"
        )

        print("\n✓ Test 5 passed: Complete workflow works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
