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
from environments.montezuma_grey import train_ppo_montezuma_grey
from environments.montezuma_grey_ram import train_ppo_montezuma_grey_ram
from environments.montezuma_ram import train_ppo_montezuma_ram
from environments.montezuma_rgb import train_ppo_montezuma_rgb
from environments.montezuma_rgb_ram import train_ppo_montezuma_rgb_ram


def test_all_environments():
    """Test all environment configurations with minimal steps."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping tests")

    environments = [
        ("RGB", train_ppo_montezuma_rgb),
        ("RAM", train_ppo_montezuma_ram),
        ("RGB-RAM", train_ppo_montezuma_rgb_ram),
        ("Grey", train_ppo_montezuma_grey),
        ("Grey-RAM", train_ppo_montezuma_grey_ram),
    ]

    print("\n" + "=" * 70)
    print("Testing all Montezuma environment configurations")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, train_func in environments:
        print(f"\n{'=' * 70}")
        print(f"Testing: {name}")
        print(f"{'=' * 70}")

        try:
            model, vec_env = train_func(
                device="cuda",
                timesteps=500,  # Very small for quick test
                seed=42,
                run=None,  # No wandb
                checkpoint_path=None,  # No checkpointing for quick test
                checkpoint_interval=100_000,
                shaping=None,
            )
            assert model is not None, "Model should be returned"
            assert vec_env is not None, "VecEnv should be returned"
            print(f"✓ PASSED: {name}")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {name}")
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
    test_all_environments()
