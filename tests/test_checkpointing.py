#!/usr/bin/env python3
"""Test checkpointing and resuming training functionality."""

import os
import sys
import tempfile
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import ale_py  # noqa: F401
import pytest
import torch

from environments.montezuma_ram import (
    load_and_continue_training_ram,
    train_ppo_montezuma_ram,
)


def test_checkpointing_basic():
    """Test basic checkpointing functionality."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Test: Basic checkpointing during training")
        print("=" * 70)

        # Train for a very short time with checkpointing
        model, vec_env = train_ppo_montezuma_ram(
            device="cuda",
            timesteps=2_000,
            seed=42,
            run=None,
            checkpoint_path=checkpoint_dir,
            checkpoint_interval=1_000,
            shaping=None,
        )

        assert model is not None
        assert vec_env is not None

        # Check that checkpoints were created
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
        assert len(checkpoint_files) >= 1, "At least one checkpoint should be created"

        # Check that VecNormalize stats exist
        vecnormalize_files = [
            f for f in os.listdir(checkpoint_dir) if f.endswith("_vecnormalize.pkl")
        ]
        assert len(vecnormalize_files) >= 1, (
            "At least one VecNormalize stats file should be created"
        )

        print("✓ Checkpointing test passed")


def test_resume_training():
    """Test resuming training from a checkpoint."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("Test: Resuming training from checkpoint")
        print("=" * 70)

        # Initial training
        model1, _ = train_ppo_montezuma_ram(
            device="cuda",
            timesteps=1_500,
            seed=42,
            run=None,
            checkpoint_path=checkpoint_dir,
            checkpoint_interval=1_000,
            shaping=None,
        )

        initial_timesteps = model1.num_timesteps
        print(f"Initial training completed: {initial_timesteps} timesteps")

        # Find the latest checkpoint
        checkpoint_files = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".zip")
        ]
        assert len(checkpoint_files) > 0, "No checkpoints found"
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

        # Resume training
        model2, _ = load_and_continue_training_ram(
            checkpoint_path=latest_checkpoint,
            device="cuda",
            additional_timesteps=1_000,
            seed=42,
            run=None,
            checkpoint_interval=1_000,
            save_checkpoint_path=checkpoint_dir,
            shaping=None,
        )

        final_timesteps = model2.num_timesteps
        print(f"Resumed training completed: {final_timesteps} timesteps")

        assert final_timesteps >= initial_timesteps + 1_000, (
            f"Expected at least {initial_timesteps + 1_000} timesteps, got {final_timesteps}"
        )

        print("✓ Resume training test passed")


if __name__ == "__main__":
    test_checkpointing_basic()
    test_resume_training()
