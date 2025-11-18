"""
Utility functions for setting random seeds to ensure reproducibility.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed, device="cpu", deterministic_gpu=False):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value
        device: 'cpu' or 'cuda' - affects GPU determinism settings
        deterministic_gpu: If True, enable deterministic GPU operations (slower but reproducible).
                          Default: False (optimize for speed)

    Returns:
        Actual device used ('cpu' or 'cuda') - may differ from input if GPU is unavailable
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # Set Python hash seed for dictionary ordering (must be set before imports in some cases)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # GPU seeds (if using CUDA/ROCm)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_gpu:
            # Enable deterministic operations (slower but reproducible)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Optimize for speed (non-deterministic but faster)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    elif device == "cuda":
        device = "cpu"

    return device


def reset_env_seed(env, seed):
    """
    Reset environment seed. Handles both Gymnasium and older Gym APIs.

    Args:
        env: Gymnasium/Gym environment
        seed: Integer seed value
    """
    try:
        # Gymnasium API (newer): reset() returns (obs, info) tuple
        obs, info = env.reset(seed=seed)
        return obs, info
    except TypeError:
        # Older Gym API or wrapper that doesn't support seed parameter
        try:
            env.seed(seed)
            obs = env.reset()
            return obs, None
        except AttributeError:
            # Last resort: just reset without seed (some wrappers don't support seeding)
            obs = env.reset()
            return obs, None
