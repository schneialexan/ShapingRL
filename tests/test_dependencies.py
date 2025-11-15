"""Comprehensive test suite to verify all dependencies and GPU acceleration."""

import pytest
import sys
import warnings

# Suppress pygame AVX2 warning (fallback - pygame should be built with AVX2 in flake.nix)
# If you see this warning, rebuild the nix environment: nix flake update && direnv reload
warnings.filterwarnings("ignore", message=".*avx2.*", category=RuntimeWarning)


def test_python_version():
    """Verify Python version is 3.11 or higher."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


def test_core_imports():
    """Test that all core dependencies can be imported and have versions."""
    import torch
    import gymnasium
    import numpy as np
    import matplotlib
    import pygame
    import wandb
    import stable_baselines3
    import moviepy  # type: ignore[import-untyped]

    # Actually use imports to avoid "unused import" warnings
    pygame.init()
    pygame.quit()
    assert moviepy.__version__  # type: ignore[attr-defined]

    # Verify versions exist
    assert torch.__version__
    assert gymnasium.__version__
    assert np.__version__
    assert matplotlib.__version__
    assert wandb.__version__
    assert stable_baselines3.__version__


def test_torch_gpu_rocm():
    """Test PyTorch ROCm (AMD GPU) availability and basic operations."""
    import torch

    # Check if ROCm is available
    has_rocm = torch.cuda.is_available()  # PyTorch uses 'cuda' API for ROCm too

    if has_rocm:
        print(f"\n✓ ROCm/GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  HIP version: {torch.version.hip}")

        # Test basic GPU tensor operations
        device = torch.device("cuda")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)

        assert z.device.type == "cuda"
        assert z.shape == (100, 100)
        print("  ✓ Basic GPU tensor operations working")

        # Test element-wise operations
        a = torch.randn(1000, device=device)
        b = torch.randn(1000, device=device)
        c = a + b
        d = torch.relu(a)
        e = torch.sigmoid(b)

        assert all(t.device.type == "cuda" for t in [c, d, e])
        print("  ✓ Element-wise GPU operations working")

        # Note: Complex neural network layers may have limited support
        # on newer GPU architectures due to library compatibility in nixpkgs
        print("  Note: For full neural network support, consider using")
        print("        CPU training or non-Nix PyTorch installation")
    else:
        pytest.skip("No ROCm/GPU detected - skipping GPU tests")


def test_numpy_operations():
    """Test NumPy basic operations."""
    import numpy as np

    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15

    matrix = np.random.rand(10, 10)
    assert matrix.shape == (10, 10)


def test_matplotlib_plotting():
    """Test Matplotlib can create plots."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    assert fig is not None
    plt.close(fig)


def test_gymnasium_environment():
    """Test Gymnasium RL environment creation and basic functionality."""
    import gymnasium as gym
    import numpy as np

    # Test classic control environment
    env = gym.make("CartPole-v1")
    observation, info = env.reset()

    assert observation is not None
    assert isinstance(observation, np.ndarray)

    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)

    env.close()


def test_ale_atari_environment():
    """Test ALE (Atari Learning Environment) registration."""
    import gymnasium as gym
    import ale_py

    # Register ALE environments
    gym.register_envs(ale_py)

    # Check that Atari environments are registered
    envs = list(gym.envs.registry.keys())
    atari_envs = [e for e in envs if "ALE" in e]

    assert len(atari_envs) > 0, "No Atari environments found"
    print(f"\n✓ {len(atari_envs)} Atari environments registered")
    print(f"  Sample: {atari_envs[:3]}")

    # Note: Actually loading environments requires ROM files
    # ROMs not included by default for licensing reasons
    # To use: Install AutoROM or manually provide ROM files


def test_pygame_functionality():
    """Test Pygame can initialize and perform basic operations."""
    import pygame

    # Initialize pygame without display
    pygame.init()
    assert pygame.get_init()

    # Test creating a surface
    surf = pygame.Surface((100, 100))
    assert surf.get_size() == (100, 100)

    # Test drawing operations
    pygame.draw.rect(surf, (255, 0, 0), (10, 10, 20, 20))
    color = surf.get_at((15, 15))
    assert color[:3] == (255, 0, 0)

    pygame.quit()


def test_torch_neural_network():
    """Test PyTorch neural network creation and forward pass."""
    import torch
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleNet()
    x = torch.randn(32, 10)  # Batch of 32 samples
    output = model(x)

    assert output.shape == (32, 2)


def test_wandb_functionality():
    """Test Weights & Biases can be initialized and used."""
    import wandb
    import os

    # Disable wandb for testing (no actual run)
    os.environ["WANDB_MODE"] = "disabled"

    # Test wandb can be initialized
    run = wandb.init(mode="disabled", project="test")
    assert run is not None

    # Test logging
    wandb.log({"test_metric": 42})
    wandb.log({"test_metric_2": 3.14})

    # Test config
    wandb.config.test_param = "test_value"
    assert wandb.config.test_param == "test_value"

    wandb.finish()


def test_stable_baselines3_functionality():
    """Test Stable Baselines3 can create agents."""
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env

    # Create a simple environment
    env = make_vec_env("CartPole-v1", n_envs=1)

    # Test PPO creation (use CPU to avoid GPU warning for MlpPolicy)
    ppo_model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    assert ppo_model is not None

    # Test DQN creation (use CPU to avoid GPU warning for MlpPolicy)
    dqn_model = DQN("MlpPolicy", env, verbose=0, device="cpu")
    assert dqn_model is not None

    env.close()


def test_imageio_functionality():
    """Test imageio can read/write images."""
    import imageio.v2 as imageio  # type: ignore[import-untyped]
    import numpy as np
    import tempfile
    import os

    # Create a test image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Test writing and reading
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        imageio.imwrite(tmp_path, img)
        assert os.path.exists(tmp_path)

        loaded_img = imageio.imread(tmp_path)
        assert loaded_img.shape == img.shape
        assert np.array_equal(loaded_img, img)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_pillow_functionality():
    """Test Pillow (PIL) image processing."""
    from PIL import Image
    import numpy as np

    # Create a test image
    img = Image.new("RGB", (100, 100), color="red")
    assert img.size == (100, 100)
    assert img.mode == "RGB"

    # Test image operations
    resized = img.resize((50, 50))
    assert resized.size == (50, 50)

    # Test conversion
    gray = img.convert("L")
    assert gray.mode == "L"

    # Test numpy conversion
    img_array = np.array(img)
    assert img_array.shape == (100, 100, 3)


def test_development_tools():
    """Test that development tools are available."""
    import subprocess
    import shutil

    # Test ruff - in Nix, ruff is a standalone binary, not a Python module
    ruff_path = shutil.which("ruff")
    if ruff_path:
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True)
        assert result.returncode == 0, "Ruff should be available"
    else:
        # Fallback: try as Python module (for non-Nix environments)
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True
        )
        assert result.returncode == 0, "Ruff should be available"

    # Test pytest (we're running in pytest, so it's available)
    import pytest

    assert pytest.__version__


def test_opencv_functionality():
    """Test OpenCV image processing operations."""
    import cv2  # type: ignore[import-untyped]
    import numpy as np

    # Create a test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 255, 255]  # White square

    # Test color conversion
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8

    # Test drawing operations
    cv2.rectangle(img, (10, 10), (20, 20), (0, 255, 0), 2)
    cv2.circle(img, (50, 50), 10, (255, 0, 0), -1)

    # Test image operations
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    assert blurred.shape == img.shape

    # Test threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    assert thresh.shape == gray.shape


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
