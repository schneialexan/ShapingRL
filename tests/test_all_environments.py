#!/usr/bin/env python3
"""Quick test to verify all Montezuma environment configurations work."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import ale_py  # noqa: F401
import pytest
import torch
from training.montezuma import train_ppo_montezuma


def test_all_observation_types():
    """Test all observation types with minimal steps."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping tests")

    observation_types = ["default", "grey", "ram", "ram-only", "grey-ram"]
    include_noop_options = [True, False]

    print("\n" + "=" * 70)
    print("Testing all Montezuma environment configurations")
    print("=" * 70)

    passed = 0
    failed = 0

    for obs_type in observation_types:
        for include_noop in include_noop_options:
            print(f"\n{'=' * 70}")
            print(f"Testing: observation_type={obs_type}, include_noop={include_noop}")
            print(f"{'=' * 70}")

            try:
                model, vec_env = train_ppo_montezuma(
                    device="cuda",
                    timesteps=500,  # Very small for quick test
                    seed=42,
                    run=None,  # No wandb
                    observation_type=obs_type,
                    include_noop=include_noop,
                    checkpoint_path=None,  # No checkpointing for quick test
                    checkpoint_interval=100_000,
                    enable_distance_reward=False,
                    distance_reward_scale=0.1,
                )
                assert model is not None, "Model should be returned"
                assert vec_env is not None, "VecEnv should be returned"
                print(f"✓ PASSED: {obs_type} with {'NOOP' if include_noop else 'no NOOP'}")
                passed += 1
            except Exception as e:
                print(f"✗ FAILED: {obs_type} with {'NOOP' if include_noop else 'no NOOP'}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print("\n" + "=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        pytest.fail(f"{failed} test(s) failed")


if __name__ == "__main__":
    test_all_observation_types()

