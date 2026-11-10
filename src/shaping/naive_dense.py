import gymnasium as gym
import numpy as np

class NaiveDenseWrapper(gym.Wrapper):
    def __init__(self, env, naive_method=1):
        super().__init__(env)
        self.last_shaping = 0.0
        self.t = 0
        match naive_method:
            case 1:
                self.naive_method = self.naive1
            case 2:
                self.naive_method = self.naive2
            case 3:
                self.naive_method = self.naive3

    def reset(self, **kwargs):
        self.t = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaping_reward = self.naive_method(obs)
        total_reward = reward + shaping_reward
        self.last_shaping = shaping_reward

        # Log detailed reward composition
        info['env_reward'] = reward
        info['shaping_reward'] = shaping_reward
        info['total_reward'] = total_reward

        self.t += 1
        return obs, total_reward, terminated, truncated, info

    # ---------------------------------------------------------
    # 1) Force going left first, then right
    # Encourage going left (negative velocity) by rewarding -velocity at start, then encourage going right.
    # If velocity is -0.05, reward is +0.05.
    # If velocity is +0.05, reward is -0.05.
    # ---------------------------------------------------------
    def naive1(self, obs):
        position, velocity = obs
        switch_step = 40  # Number of steps to enforce "go left"
        scaling = 2 # Scaled for more impact

        if self.t < switch_step:
            return -velocity * scaling
        else:
            return velocity * scaling
    
    def naive2(self, obs):
        """
        Shaping reward based on passing checkpoints in order:
        1. Slightly right from start (cp1)
        2. Slightly left (cp2)
        3. High on the right (cp3, near goal)
        Reward is given when agent moves closer to the next checkpoint in order.
        """
        position, velocity = obs
        # Define checkpoints
        cp1 = -0.4
        cp2 = -0.6
        cp3 = 0.3
        checkpoints = [cp1, cp2, cp3]
        # Initialize tracking variable
        if not hasattr(self, 'last_checkpoint'):
            self.last_checkpoint = -1  # -1 means no checkpoint passed yet
        # Determine the next checkpoint to target
        next_cp_index = self.last_checkpoint + 1
        if next_cp_index >= len(checkpoints):
            next_cp_index = len(checkpoints) - 1  # stay on last checkpoint
        next_cp = checkpoints[next_cp_index]
        # Shaping reward: negative distance to next checkpoint
        reward = -abs(position - next_cp)
        # Update checkpoint if reached
        # Consider "reached" if within a small epsilon
        if abs(position - next_cp) < 0.05:
            self.last_checkpoint = next_cp_index
        return reward 


    def naive3(self, obs):
        """
        Naive shaping reward that encourages making progress toward the goal.
        Reward is proportional to position improvement since last step.
        """
        position, velocity = obs
        scaling = 2

        # Initialize previous position tracking
        if not hasattr(self, 'last_position'):
            self.last_position = position

        # Reward = change in position (progress toward the goal)
        reward = position - self.last_position

        # Update last_position for next step
        self.last_position = position
        return reward * scaling
