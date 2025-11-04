import gymnasium as gym

class NaiveDenseWrapper(gym.Wrapper):
    def __init__(self, env, shaping_coef=0.1):
        super().__init__(env)
        self.shaping_coef = shaping_coef
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
        # Example: encourage moving to the right
        return self.shaping_coef * obs[0]
