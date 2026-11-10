import gymnasium as gym
import numpy as np

class PotentialBasedWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for Potential-Based Reward Shaping (PBRS).
    
    This modifies the reward as:
        r'_t = r_t + F(s_t, a_t, s_{t+1})
    where
        F(s, a, s') = γ * Φ(s') - Φ(s)
        s' = next state
        s = current state
    ensuring policy invariance.
    """
    def __init__(self, env, gamma=0.99, potential_scale=1.0, potential_method=2):
        super().__init__(env)
        self.gamma = gamma
        self.potential_scale = potential_scale
        self.last_potential = None
        match potential_method:
            case 1:
                self.potential_method = self.potential1
            case 2:
                self.potential_method = self.potential2
            case 3:
                self.potential_method = self.potential3

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Φ(s')
        phi_next = self.potential_method(obs)

        shaping_reward = 0.0
        if self.last_potential is not None:
            # F(s, a, s')  = γ          * Φ(s')    - Φ(s)
            shaping_reward = self.gamma * phi_next - self.last_potential
        # r'_t       = r_t    + F(s_t, a_t, s_{t+1})
        total_reward = reward + shaping_reward
        self.last_potential = phi_next

        info.update({
            'env_reward': reward,
            'shaping_reward': shaping_reward,
            'total_reward': total_reward
        })

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_potential = self.potential_method(obs)
        return obs, info

    def potential1(self, obs):
        position, velocity = obs
        return position
    
    def potential2(self, obs):
        position, velocity = obs
        return np.sin(3*position)
    
    def potential3(self, obs):
        position, velocity = obs
        return (np.sin(3*position))+((1/2)*velocity**2)
