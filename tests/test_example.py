"""Comprehensive test suite to verify all dependencies and GPU acceleration."""

import pytest
import sys


def test_python_version():
    """Verify Python version is 3.11 or higher."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


def test_core_imports():
    """Test that all core dependencies can be imported."""
    import torch
    import gymnasium
    import numpy as np
    import pandas as pd
    import scipy
    import matplotlib
    import seaborn
    import pygame
    import tqdm
    import yaml
    
    assert torch.__version__
    assert gymnasium.__version__
    assert np.__version__
    assert pd.__version__


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
        print(f"  ✓ Basic GPU tensor operations working")
        
        # Test element-wise operations
        a = torch.randn(1000, device=device)
        b = torch.randn(1000, device=device)
        c = a + b
        d = torch.relu(a)
        e = torch.sigmoid(b)
        
        assert all(t.device.type == "cuda" for t in [c, d, e])
        print(f"  ✓ Element-wise GPU operations working")
        
        # Note: Complex neural network layers may have limited support
        # on newer GPU architectures due to library compatibility in nixpkgs
        print(f"  Note: For full neural network support, consider using")
        print(f"        CPU training or non-Nix PyTorch installation")
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


def test_pandas_operations():
    """Test Pandas DataFrame operations."""
    import pandas as pd
    
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })
    
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']
    assert df['a'].sum() == 6


def test_scipy_functionality():
    """Test SciPy scientific computing."""
    import numpy as np
    from scipy import stats
    
    data = np.random.normal(0, 1, 1000)
    mean, std = stats.norm.fit(data)
    
    assert -0.5 < mean < 0.5  # Should be close to 0
    assert 0.5 < std < 1.5     # Should be close to 1


def test_matplotlib_plotting():
    """Test Matplotlib can create plots."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    
    assert fig is not None
    plt.close(fig)


def test_seaborn_plotting():
    """Test Seaborn statistical plotting."""
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    import pandas as pd
    
    df = pd.DataFrame({
        'x': range(10),
        'y': range(10)
    })
    
    plot = sns.scatterplot(data=df, x='x', y='y')
    assert plot is not None
    import matplotlib.pyplot as plt
    plt.close()


def test_gymnasium_environment():
    """Test Gymnasium RL environment creation and basic functionality."""
    import gymnasium as gym
    import numpy as np
    
    # Test classic control environment
    env = gym.make('CartPole-v1')
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
    atari_envs = [e for e in envs if 'ALE' in e]
    
    assert len(atari_envs) > 0, "No Atari environments found"
    print(f"\n✓ {len(atari_envs)} Atari environments registered")
    print(f"  Sample: {atari_envs[:3]}")
    
    # Note: Actually loading environments requires ROM files
    # ROMs not included by default for licensing reasons
    # To use: Install AutoROM or manually provide ROM files


def test_pygame_initialization():
    """Test Pygame can initialize for rendering."""
    import pygame
    
    # Initialize pygame without display
    pygame.init()
    assert pygame.get_init()
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


def test_development_tools():
    """Test that development tools are available."""
    import black
    import flake8
    
    # Just verify imports work
    assert black.__version__
    assert flake8.__version__


def test_yaml_configuration():
    """Test YAML parsing for configuration files."""
    import yaml
    
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
    
    # Dump and load
    yaml_str = yaml.dump(config)
    loaded_config = yaml.safe_load(yaml_str)
    
    assert loaded_config == config


def test_tqdm_progress():
    """Test tqdm progress bar functionality."""
    from tqdm import tqdm
    
    items = list(range(10))
    count = 0
    
    for item in tqdm(items, disable=True):  # Disable output in tests
        count += 1
    
    assert count == 10  # Should have processed all items


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])