import gymnasium as gym

class PotentialBasedWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for Potential-Based Reward Shaping (PBRS).
    
    This modifies the reward as:
        r'_t = r_t + F(s_t, a_t, s_{t+1})
    where
        F(s, a, s') = γ * Φ(s') - Φ(s)
    ensuring policy invariance.
    """
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

        info.update({
            'env_reward': reward,
            'shaping_reward': shaping_reward,
            'total_reward': total_reward
        })

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_potential = self.potential(obs)
        return obs, info
    
    def potential(self, obs):
        position, velocity = obs
        # Potential increases with position and velocity
        return self.potential_scale * (position + 0.1 * velocity)
