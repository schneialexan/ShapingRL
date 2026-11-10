import gymnasium as gym

class PotentialBasedWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99, potential_scale=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.potential_scale = potential_scale
        self.last_potential = None
        self.last_shaping = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = self.potential(obs)

        shaping_reward = 0.0
        if self.last_potential is not None:
            shaping_reward = self.gamma * phi_next - self.last_potential

        total_reward = reward + shaping_reward
        self.last_potential = phi_next
        self.last_shaping = shaping_reward

        # Log detailed reward composition
        info['env_reward'] = reward
        info['shaping_reward'] = shaping_reward
        info['total_reward'] = total_reward

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_potential = self.potential(obs)
        return obs, info
    
    def potential(self, obs):
        position, velocity = obs
        goal_pos = 0.5

        # Potential function based on position and velocity
        # Smoothly increases as the car approaches the goal and builds momentum
        distance_to_goal = goal_pos - position
        phi = self.potential_scale * (-distance_to_goal + 0.1 * velocity)
        return phi
