import gymnasium as gym
import numpy as np

class MontezumaPotentialBasedWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99, debugging=False):
        super().__init__(env)
        self.gamma = gamma
        self.debugging = debugging
        self.last_potential = None
        self.last_shaping = 0.0
        self._key_coords = (112,16) # Determined statically (for level 1)
        self._obs_dimensions = None

    @staticmethod
    def _prRed(s): print("\033[91m {}\033[00m".format(s), end="")
    @staticmethod
    def _prGreen(s): print("\033[92m {}\033[00m".format(s), end="")
    @staticmethod
    def _console_log_reward(reward:float, shaping_reward:float):
        print("Reward: ", end="")
        if reward >= 0:
            MontezumaPotentialBasedWrapper._prGreen(f"{reward:.3f}")
        else:
            MontezumaPotentialBasedWrapper._prRed(f"{reward:.3f}")

        print(" Shaping Reward: ", end="")
        if shaping_reward >= 0:
            MontezumaPotentialBasedWrapper._prGreen(f"{shaping_reward:.3f}")
        else:
            MontezumaPotentialBasedWrapper._prRed(f"{shaping_reward:.3f}")

        print() # new line 

    def _get_rgb_position(self, obs, rgb_of_object) -> tuple[float, float]:
        # code mostly adopted from coach.py
        matches = np.all(rgb_of_object == obs, axis=-1)
        indices = np.argwhere(matches)

        if len(indices) > 0:
            v, u = indices[0]
            return v, u

        diff = obs - rgb_of_object
        dist_squared = np.sum(diff ** 2, axis=-1)

        index_min = np.unravel_index(np.argmin(dist_squared), dist_squared.shape)
        v, u = index_min

        return v, u
    
    def _get_dimensions(self, obs) -> tuple[int, int]:
        return obs.shape[0], obs.shape[1]

    def _manhattan_distance(self, pos1, pos2, obs_dimensions, beta_x=0.5):
        """
        Weighted Manhattan distance.
        beta_x weights the reward for x-axis, which can be used if one axis is more important than the other.
        beta_y is (1 - beta_x). Note that we use y,x coords here. 
        """
        return ((1 - beta_x) * abs(pos1[0] - pos2[0]) + beta_x * abs(pos1[1] - pos2[1])) / (obs_dimensions[0] + obs_dimensions[1])
    

    def _potential(self, obs, scaling=1) -> float:
        character_neck_rgb = np.array([228, 111, 111])
        character_coords = self._get_rgb_position(obs, character_neck_rgb)

        if not self._obs_dimensions:
            self._obs_dimensions = self._get_dimensions(obs)
        
        # Lets put emphasis on x axis (with beta_x) to reduce penalty of descending into pit
        return -self._manhattan_distance(character_coords, self._key_coords, self._obs_dimensions, 0.7) * scaling


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = self._potential(obs)

        shaping_reward = 0.0
        if self.last_potential is not None:
            shaping_reward = self.gamma * phi_next - self.last_potential
        
        if self.debugging:
            self._console_log_reward(reward,shaping_reward)

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
        self.last_potential = self._potential(obs)
        return obs, info
