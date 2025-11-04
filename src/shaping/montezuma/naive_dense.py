import gymnasium as gym
import numpy as np


class NaiveDenseWrapper(gym.Wrapper):
    def __init__(self, env, shaping_coef=0.1):
        super().__init__(env)
        self.shaping_coef = shaping_coef
        self.last_shaping = 0.0

    def _get_approximated_character_position(self, obs) -> tuple[float, float]:
        """
        Returns v and u, where v is the number of vertical pixels from the top left corner of the image
        and u the number of horizontal pixels from the top left corner
        """
        character_neck_rgb = np.array([228, 111, 111])

        # Find positions where all 3 channels match
        matches = np.all(character_neck_rgb == obs, axis=-1)

        # Compute squared Euclidean distance for all pixels
        diff = obs - character_neck_rgb
        dist_squared = np.sum(diff ** 2, axis=-1)

        # Find the index of the minimum distance
        index_min = np.unravel_index(np.argmin(dist_squared), dist_squared.shape)

        # Get the indices
        # indices = np.argwhere(matches)

        v, u = index_min

        return v, u

    def _get_approximated_character_coords(self, obs):
        v, u = self._get_approximated_character_position(obs)
        # obs.shape =  (210, 160, 3)

        game_image_frame_height = 210
        game_image_frame_width = 160

        x = u
        y = game_image_frame_height - v

        return x, y

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaping_reward = self._shaping(obs)

        x, y = self._get_approximated_character_coords(obs)
        # print(f'{x = } {y = }')

        total_reward = (
            reward
            # + shaping_reward
        )
        self.last_shaping = shaping_reward

        #  rgb(219 106 106)

        # Log detailed reward composition
        info['env_reward'] = reward
        info['shaping_reward'] = shaping_reward
        info['total_reward'] = total_reward

        return obs, total_reward, terminated, truncated, info

    def _shaping(self, obs):
        # Example: encourage moving to the right
        return self.shaping_coef * obs[0]
