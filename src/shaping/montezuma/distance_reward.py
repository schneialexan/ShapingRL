#!/usr/bin/env python3
"""Reward shaping wrapper that rewards moving right and reaching new rooms."""

import gymnasium as gym


class DistanceRewardWrapper(gym.Wrapper):
    """
    Reward shaping wrapper that encourages the agent to:
    1. Move right (reward based on x-position increase)
    2. Reach new rooms (bonus for room transitions)
    3. Potentially jump over the first gap (bonus for significant x-movement with jump)

    Uses RAM to extract player position and room information.
    """

    def __init__(
        self,
        env,
        distance_reward_scale=0.1,
        room_bonus=50.0,
        max_x_reward_threshold=200,
        debugging=False,
    ):
        """
        Args:
            env: The environment to wrap
            distance_reward_scale: Scale factor for distance-based reward (reward per pixel moved right)
            room_bonus: Bonus reward for entering a new room
            max_x_reward_threshold: Maximum x-position to consider for distance reward (prevents overflow)
            debugging: If True, print debug information
        """
        super().__init__(env)
        self.distance_reward_scale = distance_reward_scale
        self.room_bonus = room_bonus
        self.max_x_reward_threshold = max_x_reward_threshold
        self.debugging = debugging

        # Track state
        self.last_x_position = None
        self.last_room_id = None
        self.max_x_reached = 0
        self.rooms_visited = set()

        # RAM indices (from MontezumasRAMInformationExtractor)
        self._ram_indices = {
            "player_x": 42,
            "player_y": 43,
            "room_state": 62,
        }

    def _get_ram(self):
        """Get RAM state from environment."""
        return self.env.unwrapped.ale.getRAM()

    def _get_player_x(self):
        """Get player x-position from RAM."""
        ram = self._get_ram()
        return int(ram[self._ram_indices["player_x"]])

    def _get_room_id(self):
        """Get current room ID from RAM."""
        ram = self._get_ram()
        return int(ram[self._ram_indices["room_state"]])

    def _calculate_shaping_reward(self, x_position, room_id):
        """Calculate shaping reward based on position and room."""
        shaping_reward = 0.0

        # 1. Distance-based reward: reward for moving right
        if self.last_x_position is not None:
            x_delta = x_position - self.last_x_position

            # Only reward positive movement (moving right)
            if x_delta > 0:
                # Clip x_position to prevent overflow rewards
                if x_position <= self.max_x_reward_threshold:
                    shaping_reward += x_delta * self.distance_reward_scale
                else:
                    # Still reward movement, but at reduced rate beyond threshold
                    shaping_reward += x_delta * self.distance_reward_scale * 0.1

            # Bonus for reaching new maximum x-position
            if x_position > self.max_x_reached:
                max_delta = x_position - self.max_x_reached
                self.max_x_reached = x_position
                # Small bonus for reaching new maximum
                shaping_reward += max_delta * self.distance_reward_scale * 0.5

        # 2. Room transition bonus
        if self.last_room_id is not None and room_id != self.last_room_id:
            if room_id not in self.rooms_visited:
                # First time entering this room - big bonus!
                shaping_reward += self.room_bonus
                self.rooms_visited.add(room_id)
                if self.debugging:
                    print(
                        f"🎉 New room reached! Room {room_id}, bonus: {self.room_bonus}"
                    )

        return shaping_reward

    def step(self, action):
        """Step environment and add distance-based reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get current state from RAM
        x_position = self._get_player_x()
        room_id = self._get_room_id()

        # Calculate shaping reward
        shaping_reward = self._calculate_shaping_reward(x_position, room_id)

        # Update state
        self.last_x_position = x_position
        self.last_room_id = room_id

        # Combine rewards
        total_reward = reward + shaping_reward

        # Log detailed reward composition
        info["env_reward"] = reward
        info["shaping_reward"] = shaping_reward
        info["total_reward"] = total_reward
        info["player_x"] = x_position
        info["room_id"] = room_id
        info["max_x_reached"] = self.max_x_reached

        if self.debugging:
            print(
                f"Step: x={x_position}, room={room_id}, "
                f"env_reward={reward:.2f}, shaping={shaping_reward:.2f}, "
                f"total={total_reward:.2f}"
            )

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and tracking state."""
        obs, info = self.env.reset(**kwargs)

        # Initialize state
        x_position = self._get_player_x()
        room_id = self._get_room_id()

        self.last_x_position = x_position
        self.last_room_id = room_id
        self.max_x_reached = x_position
        self.rooms_visited = {room_id}  # Start room is visited

        if self.debugging:
            print(f"Reset: x={x_position}, room={room_id}")

        return obs, info
