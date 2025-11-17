#!/usr/bin/env python3
"""Unified Montezuma's Revenge environment with configurable observation and action spaces."""

import ale_py  # noqa: F401 - Required for ALE environment registration
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from info.montezumas_info import MontezumaActions


class ActionSpaceWrapper(gym.ActionWrapper):
    """Reduce action space with optional NOOP."""

    def __init__(self, env, include_noop=True):
        super().__init__(env)
        if include_noop:
            # 7 actions: NOOP, UP, RIGHT, LEFT, DOWN, RIGHTFIRE, LEFTFIRE
            self.action_subset = [
                MontezumaActions.NOOP.value,
                MontezumaActions.UP.value,
                MontezumaActions.RIGHT.value,
                MontezumaActions.LEFT.value,
                MontezumaActions.DOWN.value,
                MontezumaActions.RIGHTFIRE.value,
                MontezumaActions.LEFTFIRE.value,
            ]
            self._action_mapping = {
                0: MontezumaActions.NOOP.value,
                1: MontezumaActions.UP.value,
                2: MontezumaActions.RIGHT.value,
                3: MontezumaActions.LEFT.value,
                4: MontezumaActions.DOWN.value,
                5: MontezumaActions.RIGHTFIRE.value,
                6: MontezumaActions.LEFTFIRE.value,
            }
        else:
            # 6 actions: UP, RIGHT, LEFT, DOWN, RIGHTFIRE, LEFTFIRE (no NOOP)
            self.action_subset = [
                MontezumaActions.UP.value,
                MontezumaActions.RIGHT.value,
                MontezumaActions.LEFT.value,
                MontezumaActions.DOWN.value,
                MontezumaActions.RIGHTFIRE.value,
                MontezumaActions.LEFTFIRE.value,
            ]
            self._action_mapping = {
                0: MontezumaActions.UP.value,
                1: MontezumaActions.RIGHT.value,
                2: MontezumaActions.LEFT.value,
                3: MontezumaActions.DOWN.value,
                4: MontezumaActions.RIGHTFIRE.value,
                5: MontezumaActions.LEFTFIRE.value,
            }
        self.action_space = spaces.Discrete(len(self.action_subset))

    def action(self, act):
        return self._action_mapping[act]


class GrayscaleObservationWrapper(gym.Wrapper):
    """Convert RGB observations to grayscale."""

    def __init__(self, env):
        super().__init__(env)
        # Update observation space: (210, 160, 3) -> (210, 160, 1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 1), dtype=np.uint8
        )

    def _convert_to_grayscale(self, obs):
        """Convert RGB to grayscale using standard weights."""
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            grayscale = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            return np.expand_dims(grayscale, axis=2)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_to_grayscale(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_to_grayscale(obs), info


class RAMObservationWrapper(gym.Wrapper):
    """Add RAM features to observations (dictionary mode)."""

    def __init__(self, env):
        super().__init__(env)
        self._ram_indices = {
            "room_number": 3,
            "player_x": 42,
            "player_y": 43,
            "player_direction": 52,
            "enemy_skull_x": 47,
            "enemy_skull_y": 46,
            "key_monster_x": 44,
            "key_monster_y": 45,
            "level": 57,
            "num_lives": 58,
            "items_in_inventory_count": 61,
            "room_state": 62,
        }
        num_features = len(self._ram_indices)  # 12 features

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
                ),
                "ram_features": spaces.Box(
                    low=0, high=255, shape=(num_features,), dtype=np.uint8
                ),
            }
        )

    def _extract_ram_features(self):
        ram = self.env.unwrapped.ale.getRAM()
        features = [int(ram[idx]) for idx in self._ram_indices.values()]
        return np.array(features, dtype=np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram_features = self._extract_ram_features()
        return (
            {"image": obs, "ram_features": ram_features},
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram_features = self._extract_ram_features()
        return {"image": obs, "ram_features": ram_features}, info


class RAMFlattenedObservationWrapper(gym.Wrapper):
    """Flatten RAM features and concatenate with image observations."""

    def __init__(self, env):
        super().__init__(env)
        self._ram_indices = {
            "room_number": 3,
            "player_x": 42,
            "player_y": 43,
            "player_direction": 52,
            "enemy_skull_x": 47,
            "enemy_skull_y": 46,
            "key_monster_x": 44,
            "key_monster_y": 45,
            "level": 57,
            "num_lives": 58,
            "items_in_inventory_count": 61,
            "room_state": 62,
        }
        num_ram_features = len(self._ram_indices)  # 12 features
        obs_size = 210 * 160 * 3 + num_ram_features
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.uint8
        )

    def _extract_ram_features(self):
        ram = self.env.unwrapped.ale.getRAM()
        features = [int(ram[idx]) for idx in self._ram_indices.values()]
        return np.array(features, dtype=np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram_features = self._extract_ram_features()
        image_flat = obs.flatten()
        obs_flat = np.concatenate([image_flat, ram_features])
        return obs_flat, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram_features = self._extract_ram_features()
        image_flat = obs.flatten()
        obs_flat = np.concatenate([image_flat, ram_features])
        return obs_flat, info


class RAMOnlyObservationWrapper(gym.Wrapper):
    """Return only RAM features, no image data."""

    def __init__(self, env):
        super().__init__(env)
        self._ram_indices = {
            "room_number": 3,
            "player_x": 42,
            "player_y": 43,
            "player_direction": 52,
            "enemy_skull_x": 47,
            "enemy_skull_y": 46,
            "key_monster_x": 44,
            "key_monster_y": 45,
            "level": 57,
            "num_lives": 58,
            "items_in_inventory_count": 61,
            "room_state": 62,
        }
        num_ram_features = len(self._ram_indices)  # 12 features
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(num_ram_features,), dtype=np.uint8
        )

    def _extract_ram_features(self):
        ram = self.env.unwrapped.ale.getRAM()
        features = [int(ram[idx]) for idx in self._ram_indices.values()]
        return np.array(features, dtype=np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram_features = self._extract_ram_features()
        return ram_features, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram_features = self._extract_ram_features()
        return ram_features, info


class GreyRAMObservationWrapper(gym.Wrapper):
    """Combine grayscale image with RAM features (flattened)."""

    def __init__(self, env):
        super().__init__(env)
        self._ram_indices = {
            "room_number": 3,
            "player_x": 42,
            "player_y": 43,
            "player_direction": 52,
            "enemy_skull_x": 47,
            "enemy_skull_y": 46,
            "key_monster_x": 44,
            "key_monster_y": 45,
            "level": 57,
            "num_lives": 58,
            "items_in_inventory_count": 61,
            "room_state": 62,
        }
        num_ram_features = len(self._ram_indices)  # 12 features
        # Grayscale: (210, 160, 1) -> 210*160 = 33600
        obs_size = 210 * 160 + num_ram_features
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.uint8
        )

    def _extract_ram_features(self):
        ram = self.env.unwrapped.ale.getRAM()
        features = [int(ram[idx]) for idx in self._ram_indices.values()]
        return np.array(features, dtype=np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram_features = self._extract_ram_features()
        # obs is grayscale (210, 160, 1), flatten it
        image_flat = obs.flatten()
        obs_flat = np.concatenate([image_flat, ram_features])
        return obs_flat, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram_features = self._extract_ram_features()
        image_flat = obs.flatten()
        obs_flat = np.concatenate([image_flat, ram_features])
        return obs_flat, info


def make_montezuma(
    render_mode="rgb_array",
    observation_type="default",  # default, grey, ram, ram-only, grey-ram
    include_noop=True,  # True = 7 actions, False = 6 actions
    enable_distance_reward=False,
    distance_reward_scale=0.1,
):
    """
    Create unified Montezuma's Revenge environment with configurable options.

    Args:
        render_mode: Rendering mode ("rgb_array", "human", or None)
        observation_type: Type of observations
            - "default": RGB image (210, 160, 3)
            - "grey": Grayscale image (210, 160, 1)
            - "ram": Dictionary with RGB image + RAM features
            - "ram-only": Only RAM features (12,)
            - "grey-ram": Flattened grayscale + RAM features
        include_noop: If True, include NOOP action (7 actions), else 6 actions
        enable_distance_reward: If True, add reward shaping for moving right
        distance_reward_scale: Scale factor for distance reward (default: 0.1)

    Returns:
        Configured Montezuma's Revenge environment
    """
    env_name = "ALE/MontezumaRevenge-v5"

    # Check if environment is already registered
    if env_name not in [spec.id for spec in gym.envs.registry.values()]:
        register(
            id=env_name,
            entry_point="ale_py.roms:MontezumaRevenge",
            max_episode_steps=4_000,
            reward_threshold=3000.0,
        )

    # Create base environment
    env = gym.make(env_name, render_mode=render_mode)

    # 1. Reduce action space (always first)
    env = ActionSpaceWrapper(env, include_noop=include_noop)

    # 2. Apply observation wrappers based on type
    if observation_type == "default":
        # RGB image only, no additional wrapper needed
        pass
    elif observation_type == "grey":
        env = GrayscaleObservationWrapper(env)
    elif observation_type == "ram":
        env = RAMObservationWrapper(env)
    elif observation_type == "ram-only":
        env = RAMOnlyObservationWrapper(env)
    elif observation_type == "grey-ram":
        env = GrayscaleObservationWrapper(env)
        env = GreyRAMObservationWrapper(env)
    else:
        raise ValueError(
            f"Unknown observation_type: {observation_type}. "
            f"Must be one of: default, grey, ram, ram-only, grey-ram"
        )

    # 3. Add distance-based reward shaping (optional)
    if enable_distance_reward:
        from shaping.montezuma.distance_reward import DistanceRewardWrapper

        env = DistanceRewardWrapper(
            env, distance_reward_scale=distance_reward_scale, debugging=False
        )

    return env


if __name__ == "__main__":
    # Test all observation types
    print("Testing all observation types...\n")

    for obs_type in ["default", "grey", "ram", "ram-only", "grey-ram"]:
        print(f"Testing {obs_type}...")
        env = make_montezuma(observation_type=obs_type, include_noop=False)
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        obs, info = env.reset()
        if isinstance(obs, dict):
            print(f"  Observation: dict with keys {obs.keys()}")
            for k, v in obs.items():
                print(f"    {k}: shape={v.shape}")
        else:
            print(f"  Observation shape: {obs.shape}")
        env.close()
        print()

    print("All tests passed!")
