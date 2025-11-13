import logging
from collections import defaultdict

import cv2
import gymnasium as gym
import numpy as np
import wandb

from information_extraction.montezumas_revenge import MontezumasInformationExtractor, \
    MontezumasRAMInformationExtractor
from log.montezumas import configure_logging_for_montezumas
from util import PROJECT_ROOT_PATH, create_dir_if_not_exists

configure_logging_for_montezumas('test_dense_wrapper')
logger = logging.getLogger(__name__)


class TestDenseWrapper(gym.Wrapper):

    def __init__(self, env, n_grid_items_horizontal=7, n_grid_items_vertical=10):
        super().__init__(env)

        self._info_extractor = MontezumasRAMInformationExtractor(self)

        self._last_agent_pos: np.array = None
        self._last_agent_n_lives: int = 5
        self._visited_pos = defaultdict(lambda: False)

        self._n_grid_items_horizontal = n_grid_items_horizontal
        self._n_grid_items_vertical = n_grid_items_vertical

        self._horizontal_spacing = int(
            MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT / n_grid_items_vertical
        )
        self._vertical_spacing = int(
            MontezumasInformationExtractor.GAME_IMAGE_FRAME_WIDTH / n_grid_items_horizontal
        )

        self._reset()

        self._last_obs = None
        self._die_position = None

        self._step_counter = 0
        self._episode_counter = 0

        self._falling_distance = 0

    def _reset(self):
        self._last_agent_n_lives = 5
        self._visited_pos = defaultdict(lambda: False)
        self._reset_grid_reward_states()
        self._last_agent_pos: np.array = None
        self._falling_distance = 0

    def _draw_not_visited_rect(self, obs, x, y, width, height):
        # Rectangle parameters

        y = MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT - y - height

        top_left = (x, y)
        bottom_right = (x + width, y + height)
        rgba_color = (0, 0, 255, 128)  # Red with 50% opacity (alpha = 128 / 255)

        # Create overlay for transparency
        overlay = obs.copy()

        # Draw rectangle on overlay (ignore alpha for now)
        cv2.rectangle(overlay, top_left, bottom_right, rgba_color[:3], -1)

        # Blend overlay with original image using alpha
        alpha = rgba_color[3] / 255.0
        cv2.addWeighted(overlay, alpha, obs, 1 - alpha, 0, obs)

    def _save_obs(self, obs, description, debug=False, player_pos=None):
        img_dir = f'{PROJECT_ROOT_PATH}/img_debug/{wandb.run.id}/{self._episode_counter:05}_episode_{5 - self._last_agent_n_lives}_lost_lifes/'
        create_dir_if_not_exists(img_dir)

        # Define the text and position
        text = description
        position = (0, 25)  # (x, y) coordinates for the text

        # Define font, scale, color, and thickness
        font = cv2.FONT_ITALIC
        font_scale = 0.4
        color = (0, 255, 0)  # Green in BGR format
        thickness = 1

        # Add text to the image

        for key, was_visited in self._room_grid.items():
            cord_list = key.split(',')
            row = int(cord_list[0])
            col = int(cord_list[1])

            if not was_visited:
                self._draw_not_visited_rect(
                    obs=obs,
                    x=col * self._horizontal_spacing,
                    y=row * self._vertical_spacing,
                    width=self._horizontal_spacing,
                    height=self._vertical_spacing
                )

        cv2.putText(obs, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        if player_pos is not None:
            # cv2.rectangle(
            #     obs,
            #     (25, 25),
            #     (50, 50),
            #     [255, 0, 0], 1
            # )

            y = MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT - player_pos[1]
            bounding_box_width = 11
            bounding_box_height = 21
            top_left = (player_pos[0] - (bounding_box_width // 2), y - (bounding_box_height // 2))
            bottom_right = (
                int(player_pos[0] + (bounding_box_width // 2)),
                int(y + (bounding_box_height // 2))
            )

            cv2.rectangle(
                obs,
                top_left,
                bottom_right,
                [255, 0, 0], 1
            )

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'{img_dir}/{self._step_counter:05}_color_img.jpg', obs)

        if debug:
            cv2.imshow("image", obs)
            cv2.waitKey()

    def _reset_life_grid_state(self):
        self._room_life_grid = {
            f'{r},{c}': False for c in range(self._n_grid_items_horizontal + 1) for r in
            range(self._n_grid_items_vertical + 1)
        }

    def _reset_grid_reward_states(self):

        self._room_grid = {
            f'{r},{c}': False for c in range(self._n_grid_items_horizontal + 1) for r in
            range(self._n_grid_items_vertical + 1)
        }

        self._reset_life_grid_state()

        # self._n_room_grid_items = len(self._room_grid.items())
        # self._explored_all_grid_items_reward = 30
        # self._explored_new_grid_item_reward = self._explored_all_grid_items_reward / self._n_room_grid_items
        #
        # self._die_reward = -((self._explored_all_grid_items_reward / 4) / 4)

        self._n_room_grid_items = len(self._room_grid.items())
        self._explored_all_grid_items_reward = 30
        self._explored_new_grid_item_reward = 1

        self._fall_reward = -2
        self._die_reward = -3
        self._die_position = None
        # self._game_over_reward = self._die_reward * 2

    def step(self, action):
        self._step_counter += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaping_reward = 0
        agent_died = False
        is_respawning = False

        ram_player_position = self._info_extractor.get_player_position()
        ram_player_lives = self._info_extractor.get_player_number_of_lives()
        ram_room = self._info_extractor.get_room_id()

        # n_lives = MontezumasInformationExtractor.get_number_of_lives(obs)
        # agent_pos = np.array(MontezumasInformationExtractor.get_approximated_character_coords(obs))

        n_lives = ram_player_lives
        agent_pos = np.array(ram_player_position)

        #self._save_obs(obs, "test position", False, player_pos=ram_player_position)

        if self._die_position is not None:
            distance_to_die_pos = np.linalg.norm(
                agent_pos - self._die_position
            )

            if distance_to_die_pos != 0:
                self._die_position = None
                self._last_agent_pos = agent_pos
                # logger.info(f'{self._step_counter}: respawned')
                self._save_obs(obs, 'agent respawned!')
            else:
                is_respawning = True
                # logger.info(f'{self._step_counter}: respawning')

        grid_column = int(agent_pos[0] // self._horizontal_spacing)
        grid_row = int(agent_pos[1] // self._vertical_spacing)

        if n_lives < self._last_agent_n_lives:
            agent_died = True
            # logger.info(f'{self._step_counter}: agent died {self._last_agent_n_lives} => {n_lives}')
            shaping_reward += self._die_reward
        if (not self._room_grid[f'{grid_row},{grid_column}']) and (not agent_died):
            shaping_reward += self._explored_new_grid_item_reward
        elif (self._room_life_grid[f'{grid_row},{grid_column}']) and (not agent_died):
            # shaping_reward += -0.0001
            shaping_reward += -0.001

        self._room_grid[f'{grid_row},{grid_column}'] = True
        self._room_life_grid[f'{grid_row},{grid_column}'] = True

        if self._last_agent_pos is not None:
            last_agent_y = self._last_agent_pos[1]
            current_agent_y = agent_pos[1]
            if (not agent_died) and (not is_respawning):

                y_vel = abs(float(
                    current_agent_y - last_agent_y)
                )  # if falling or going down y_vel is negative

                # if (y_vel >= 8) and (current_agent_y < last_agent_y):
                if (y_vel >= 8) and (current_agent_y < last_agent_y):
                    self._falling_distance += y_vel

            if current_agent_y >= last_agent_y:
                self._falling_distance = 0

        if terminated or truncated:
            self._reset()
            self._episode_counter += 1
            self._step_counter = 0

        if (self._falling_distance > 25) and (not agent_died) and (not is_respawning):
            self._falling_distance = 0
            shaping_reward += self._fall_reward
            self._save_obs(obs, 'punish falling')
            # logger.info(f'{self._step_counter}: falling more than 4 steps')

        if agent_died:
            self._last_agent_pos = None
            self._die_position = agent_pos
            self._falling_distance = 0
            self._save_obs(obs, 'agent died!')
            self._reset_life_grid_state()
        else:
            self._last_agent_pos = agent_pos

        self._last_agent_n_lives = n_lives

        total_reward = (
            reward +
            shaping_reward
        )

        info['env_reward'] = reward
        info['shaping_reward'] = shaping_reward
        info['total_reward'] = total_reward

        return np.zeros_like(obs), total_reward, terminated, truncated, info
