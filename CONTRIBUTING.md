# Contributing Guide

This guide helps team members contribute to the project effectively.

## Development Workflow

### 1. Setting Up Your Environment

Follow the setup instructions in the main [README.md](README.md) - either using Nix or pip/conda.

### 2. Creating a New Feature

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
   Branch naming conventions:
   - `feature/` - for new features
   - `fix/` - for bug fixes
   - `experiment/` - for experimental work
   - `docs/` - for documentation updates

2. **Make your changes:**
   - Write clean, readable code
   - Add docstrings to all functions and classes
   - Follow the code style guide below

3. **Test your changes:**
   ```bash
   pytest tests/
   ```

4. **Format your code:**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Clear description of your changes"
   ```
   
   Commit message conventions:
   - Start with a verb (Add, Fix, Update, Remove, etc.)
   - Keep the first line under 72 characters
   - Example: "Add potential-based reward shaping implementation"

6. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guide

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length:** 100 characters (enforced by Black)
- **Indentation:** 4 spaces (no tabs)
- **Docstrings:** Use Google-style docstrings

Example:
```python
def compute_potential_shaping(state: np.ndarray, next_state: np.ndarray, gamma: float) -> float:
    """Compute potential-based reward shaping.
    
    Args:
        state: Current state
        next_state: Next state
        gamma: Discount factor
        
    Returns:
        Shaped reward value
        
    Raises:
        ValueError: If gamma is not in [0, 1]
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    
    # Implementation here
    pass
```

### File Organization

- **One class per file** (for major components)
- **Related functions can share a file** (for utilities)
- **Keep files under 500 lines** (split if larger)

### Import Order

1. Standard library imports
2. Third-party imports
3. Local application imports

Separate each group with a blank line:

```python
import os
from typing import Dict, List

import numpy as np
import torch
import gymnasium as gym

from src.core.replay_buffer import ReplayBuffer
```

## Testing

### Writing Tests

- Place test files in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Write tests for all new functionality
- Aim for >80% code coverage

Example test:
```python
def test_replay_buffer_sampling():
    """Test that replay buffer correctly samples experiences."""
    buffer = ReplayBuffer(capacity=100)
    
    # Add some transitions
    for i in range(50):
        buffer.add(state=i, action=i, reward=i, next_state=i+1, done=False)
    
    # Sample a batch
    batch = buffer.sample(batch_size=32)
    
    assert len(batch) == 32
    assert all(isinstance(item, tuple) for item in batch)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_example.py

# Run specific test function
pytest tests/test_example.py::test_imports
```

## Documentation

### Docstrings

All public functions, classes, and methods must have docstrings:

```python
class QLearningAgent:
    """Q-Learning agent for tabular environments.
    
    This agent uses the Q-learning algorithm to learn optimal policies
    in environments with discrete state and action spaces.
    
    Attributes:
        q_table: The Q-value table
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma)
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.99):
        """Initialize the Q-Learning agent.
        
        Args:
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
        """
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
```

### Comments

- Write comments for complex logic
- Don't comment obvious code
- Use `TODO:` for future work
- Use `FIXME:` for known issues

```python
# TODO: Implement prioritized experience replay
# FIXME: Handle edge case when buffer is empty
```

## Git Best Practices

### Before Committing

- [ ] Code is formatted with Black
- [ ] No linter warnings from flake8
- [ ] Tests pass
- [ ] New tests added for new features
- [ ] Documentation updated

### Commit Messages

Good commit messages:
```
✓ Add DQN agent implementation
✓ Fix bug in replay buffer sampling
✓ Update documentation for reward shaping
✓ Refactor potential-based shaping module
```

Bad commit messages:
```
✗ Update
✗ Fix stuff
✗ WIP
✗ asdfasdf
```

### Pull Requests

When creating a pull request:

1. **Write a clear title and description**
2. **Link related issues** (if any)
3. **Request review from team members**
4. **Address review comments**
5. **Ensure CI passes** (when set up)

## Project-Specific Guidelines

### Experiment Tracking

When running experiments:

1. **Create a notebook** in `notebooks/` with a descriptive name
2. **Document hyperparameters** clearly
3. **Save results** (plots, metrics) in the notebook
4. **Commit the notebook** with results visible

### Adding New Algorithms

When adding a new RL algorithm:

1. **Create a new file** in `src/agents/`
2. **Follow the agent interface** (if one exists)
3. **Add comprehensive docstrings**
4. **Include usage examples** in docstring or notebook
5. **Add tests** for core functionality

### Adding New Environments

When adding a new environment or wrapper:

1. **Create a new file** in `src/environments/`
2. **Follow Gymnasium's interface**
3. **Document the observation/action spaces**
4. **Include example usage**

## Questions?

If you have questions about contributing:

1. Check this guide first
2. Check the main README
3. Ask in the team chat
4. Create an issue for discussion

## Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Original Reward Shaping Paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
