#!/usr/bin/env python3
"""Manhattan distance-based potential reward shaping for Montezuma's Revenge.

Uses potential-based reward shaping with Manhattan distance between player and key.
Mathematical formulation:
    φ(s) = -d_manhattan((player_x, player_y), (key_x, key_y)) + room_bonus * n_visited_rooms(s)
    r_shaping = γ * φ(s') - φ(s)

where:
    - d_manhattan(p1, p2) = |p1_x - p2_x| + |p1_y - p2_y|
    - n_visited_rooms(s) is the number of unique rooms visited up to state s
    - γ is the discount factor

The potential function φ(s) depends ONLY on the current state s, making this
mathematically valid potential-based reward shaping that preserves optimal policies.
"""

import gymnasium as gym


class ManhattanPotentialWrapper(gym.Wrapper):
    """
    Potential-based reward shaping using Manhattan distance between player and key.

    The potential function φ(s) is defined as:
        φ(s) = -d_manhattan((player_x, player_y), (key_x, key_y)) + room_bonus * n_visited_rooms(s)

    where:
        - d_manhattan(p1, p2) = |p1_x - p2_x| + |p1_y - p2_y|
        - n_visited_rooms(s) is the number of unique rooms visited up to state s
        - All state information (player position, key position, room_id) comes from RAM

    Reward shaping formula:
        r_shaping = γ * φ(s') - φ(s)

    This is mathematically valid potential-based reward shaping because φ(s) depends
    only on the current state s (all information is available from RAM in the state).
    """

    def __init__(
        self,
        env,
        gamma=0.99,
        room_bonus=50.0,
        debugging=False,
    ):
        """
        Args:
            env: The environment to wrap
            gamma: Discount factor for potential-based reward shaping
            room_bonus: Bonus value added to potential for each unique room visited
            debugging: If True, print debug information
        """
        super().__init__(env)
        self.gamma = gamma
        self.room_bonus = room_bonus
        self.debugging = debugging

        # Track state for potential-based shaping
        # Note: rooms_visited is part of the state we track to compute n_visited_rooms(s)
        self.last_potential = None
        self.rooms_visited = set()

        # RAM indices (from MontezumasRAMInformationExtractor)
        self._ram_indices = {
            "player_x": 42,
            "player_y": 43,
            "key_x": 44,
            "key_y": 45,
            "room_state": 62,
        }

    def _get_ram(self):
        """Get RAM state from environment."""
        return self.env.unwrapped.ale.getRAM()

    def _get_player_position(self) -> tuple[int, int]:
        """Get player (x, y) position from RAM."""
        ram = self._get_ram()
        player_x = int(ram[self._ram_indices["player_x"]])
        player_y = int(ram[self._ram_indices["player_y"]])
        return player_x, player_y

    def _get_key_position(self) -> tuple[int, int]:
        """Get key (x, y) position from RAM."""
        ram = self._get_ram()
        key_x = int(ram[self._ram_indices["key_x"]])
        key_y = int(ram[self._ram_indices["key_y"]])
        return key_x, key_y

    def _get_room_id(self) -> int:
        """Get current room ID from RAM."""
        ram = self._get_ram()
        return int(ram[self._ram_indices["room_state"]])

    def _manhattan_distance(
        self, pos1: tuple[int, int], pos2: tuple[int, int]
    ) -> float:
        """
        Calculate Manhattan distance between two positions.

        Mathematical definition:
            d_manhattan(p1, p2) = |p1_x - p2_x| + |p1_y - p2_y|

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)

        Returns:
            Manhattan distance between pos1 and pos2
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _potential(self) -> float:
        """
        Calculate potential function φ(s) for current state.

        Mathematical definition:
            φ(s) = -d_manhattan((player_x, player_y), (key_x, key_y)) + room_bonus * n_visited_rooms(s)

        where:
            - d_manhattan(p1, p2) = |p1_x - p2_x| + |p1_y - p2_y|
            - n_visited_rooms(s) = |rooms_visited| (number of unique rooms visited)

        The potential function depends ONLY on the current state s:
            - player_x, player_y: from RAM (part of state)
            - key_x, key_y: from RAM (part of state)
            - room_id: from RAM (part of state)
            - n_visited_rooms: tracked state (number of unique rooms visited)

        The negative sign on distance ensures that moving closer to the key
        (reducing distance) increases the potential, resulting in positive shaping reward.
        The room_bonus term encourages exploration of new rooms.

        Returns:
            Potential value φ(s) = -d_manhattan(player, key) + room_bonus * n_visited_rooms
        """
        player_pos = self._get_player_position()
        key_pos = self._get_key_position()
        distance = self._manhattan_distance(player_pos, key_pos)

        # n_visited_rooms(s) = number of unique rooms visited
        n_visited_rooms = len(self.rooms_visited)

        return -distance + self.room_bonus * n_visited_rooms

    def step(self, action):
        """
        Step environment and add potential-based reward shaping.

        Reward shaping formula:
            r_shaping = γ * φ(s') - φ(s)

        where:
            - φ(s) = -d_manhattan((player_x, player_y), (key_x, key_y)) + room_bonus * n_visited_rooms(s)
            - γ is the discount factor

        This is mathematically valid potential-based reward shaping because φ(s) depends
        only on the current state s, preserving optimal policies.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update rooms_visited based on current state (room_id from RAM)
        room_id = self._get_room_id()
        self.rooms_visited.add(room_id)

        # Calculate potential for new state
        phi_next = self._potential()

        # Calculate potential-based shaping reward: γ * φ(s') - φ(s)
        shaping_reward = 0.0
        if self.last_potential is not None:
            shaping_reward = self.gamma * phi_next - self.last_potential

        # Update state
        self.last_potential = phi_next

        # Combine rewards
        total_reward = reward + shaping_reward

        # Log detailed reward composition
        player_pos = self._get_player_position()
        key_pos = self._get_key_position()
        distance = self._manhattan_distance(player_pos, key_pos)
        n_visited_rooms = len(self.rooms_visited)

        info["env_reward"] = reward
        info["shaping_reward"] = shaping_reward
        info["total_reward"] = total_reward
        info["player_x"] = player_pos[0]
        info["player_y"] = player_pos[1]
        info["key_x"] = key_pos[0]
        info["key_y"] = key_pos[1]
        info["manhattan_distance"] = distance
        info["potential"] = phi_next
        info["room_id"] = room_id
        info["n_visited_rooms"] = n_visited_rooms

        if self.debugging:
            print(
                f"Step: player=({player_pos[0]}, {player_pos[1]}), "
                f"key=({key_pos[0]}, {key_pos[1]}), "
                f"distance={distance}, "
                f"room={room_id}, "
                f"n_visited_rooms={n_visited_rooms}, "
                f"φ(s)={phi_next:.2f}, "
                f"env_reward={reward:.2f}, "
                f"shaping={shaping_reward:.2f}, "
                f"total={total_reward:.2f}"
            )

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and initialize potential-based shaping state."""
        obs, info = self.env.reset(**kwargs)

        # Initialize state
        room_id = self._get_room_id()
        self.rooms_visited = {room_id}  # Start room is visited
        self.last_potential = self._potential()

        if self.debugging:
            player_pos = self._get_player_position()
            key_pos = self._get_key_position()
            distance = self._manhattan_distance(player_pos, key_pos)
            print(
                f"Reset: player=({player_pos[0]}, {player_pos[1]}), "
                f"key=({key_pos[0]}, {key_pos[1]}), "
                f"distance={distance}, room={room_id}"
            )

        return obs, info
