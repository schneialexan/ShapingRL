import gymnasium as gym

class NaiveDenseWrapper(gym.Wrapper):
    def __init__(self, env, shaping_coef=0.1, velocity_coef=0.05, penalty_coef=0.02):
        super().__init__(env)
        self.shaping_coef = shaping_coef
        self.velocity_coef = velocity_coef
        self.penalty_coef = penalty_coef
        self.last_shaping = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaping_reward = self._shaping(obs)
        total_reward = reward + shaping_reward
        self.last_shaping = shaping_reward

        # Log detailed reward composition
        info['env_reward'] = reward
        info['shaping_reward'] = shaping_reward
        info['total_reward'] = total_reward

        return obs, total_reward, terminated, truncated, info

    def _shaping(self, obs):
        position, velocity = obs
        goal_pos = 0.5  # MountainCar goal position

        # Encourage moving toward the goal
        progress_reward = self.shaping_coef * (position / goal_pos)

        # Encourage building momentum
        velocity_bonus = self.velocity_coef * abs(velocity)

        # Penalize being too far left (bottom of valley)
        penalty = -self.penalty_coef * (abs(position + 0.5) ** 2)

        shaping = progress_reward + velocity_bonus + penalty
        return shaping
