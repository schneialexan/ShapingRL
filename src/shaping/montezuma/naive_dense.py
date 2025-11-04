import logging
from collections import deque
from copy import copy
from typing import Self, List

import gymnasium as gym
import numpy as np

from log.montezumas import configure_logging_for_montezumas

game_image_frame_height = 210
game_image_frame_width = 160

configure_logging_for_montezumas('guide_shaper')
logger = logging.getLogger(__name__)


class GuideItem:
    def __init__(self, name: str, grid: int, cell: int, row: int, n_times: int):
        self.name = name
        self.grid = grid
        self.cell = cell
        self.row = row
        self.n_times = n_times
        self._visit_count = 0

    def visit(self):
        self._visit_count += 1

    @property
    def is_completed(self) -> bool:
        return self._visit_count >= self.n_times

    @property
    def n_times_visited(self) -> float:
        return self._visit_count


class GuideBuilder:

    def __init__(self):
        self._checkpoints: deque[GuideItem] = deque()

    def set_goal_go_to(
        self,
        name: str,
        grid: int,
        row: int,
        cell: int,
        n_times: int = 1
    ) -> Self:
        self._checkpoints.append(
            GuideItem(name=name, n_times=n_times, grid=grid, cell=cell, row=row)
        )

        return self

    def then_set_goal_go_to(
        self,
        name: str,
        grid: int,
        row: int,
        cell: int,
        n_times: int = 1
    ) -> Self:
        self.set_goal_go_to(
            name=name,
            n_times=n_times,
            grid=grid,
            cell=cell,
            row=row
        )

        return self

    def build(self) -> deque[GuideItem]:
        return_checkpoints = copy(self._checkpoints)

        return return_checkpoints


class NaiveDenseWrapper(gym.Wrapper):
    def __init__(self, env, shaping_coef=0.1):
        super().__init__(env)
        self.shaping_coef = shaping_coef
        self.last_shaping = 0.0

        self._reach_guide_point_reward: float = 100

        self._character_neck_rgb = np.array([228, 111, 111])

        guide_builder = GuideBuilder()

        self._checkpoints = (guide_builder.set_goal_go_to(
            name="Right ladder top",
            grid=8,
            cell=5,
            row=4,
            n_times=100
        ).then_set_goal_go_to(
            name="Right ladder middle",
            grid=8,
            row=3,
            cell=5,
            n_times=100
        ).then_set_goal_go_to(
            name="Right ladder bottom",
            grid=8,
            row=2,
            cell=5,
            n_times=100
        ).then_set_goal_go_to(
            name="Right ladder bottom with little offset left",
            grid=8,
            row=2,
            cell=4,
            n_times=100
        ).then_set_goal_go_to(
            name="Left ladder bottom",
            grid=8,
            row=2,
            cell=1,
            n_times=100
        ).build())

        self._current_guide_item: GuideItem | None = None

        self._current_watch_top_left_x = None
        self._current_watch_top_left_y = None

        self._current_watch_bottom_left_x = None
        self._current_watch_bottom_left_y = None

        self._current_watch_top_right_x = None
        self._current_watch_top_right_y = None

        self._current_watch_bottom_right_x = None
        self._current_watch_bottom_right_y = None
        self._stop_checking_for_rewards_util_next_terminate_state = False

    def _get_shaping_reward(self, obs) -> float:

        if self._stop_checking_for_rewards_util_next_terminate_state:
            return 0

        if (self._current_guide_item is None) or self._current_guide_item.is_completed:
            self._current_guide_item = self._checkpoints.popleft()

            logger.info(f'using new guide point with name "{self._current_guide_item.name}"')

            grid_cell_width = game_image_frame_width / self._current_guide_item.grid
            grid_cell_height = game_image_frame_height / self._current_guide_item.grid

            self._current_watch_top_left_y = self._current_guide_item.row * grid_cell_height
            self._current_watch_bottom_left_y = (
                                                    self._current_guide_item.row - 1
                                                ) * grid_cell_height

            self._current_watch_top_right_y = self._current_watch_top_left_y
            self._current_watch_bottom_right_y = self._current_watch_bottom_left_y

            self._current_watch_top_left_x = (self._current_guide_item.cell - 1) * grid_cell_width
            self._current_watch_top_right_x = self._current_guide_item.cell * grid_cell_width

            self._current_watch_bottom_left_x = self._current_watch_top_left_x
            self._current_watch_bottom_right_x = self._current_watch_top_right_x

        character_neck_x, character_neck_y = self._get_approximated_character_coords(obs)

        if (
            (self._stop_checking_for_rewards_util_next_terminate_state == False) and
            (character_neck_x > self._current_watch_top_left_x) and
            (character_neck_x < self._current_watch_top_right_x) and
            (character_neck_y < self._current_watch_top_left_y) and
            (character_neck_y > self._current_watch_bottom_left_y)
        ):
            self._current_guide_item.visit()

            logger.info(
                f'Reached goal pos for "{self._current_guide_item.name}" ({self._current_guide_item.n_times_visited}/{self._current_guide_item.n_times})'
            )

            self._stop_checking_for_rewards_util_next_terminate_state = True
            return self._reach_guide_point_reward
        elif (
            (self._stop_checking_for_rewards_util_next_terminate_state == True) and
            (character_neck_x > self._current_watch_top_left_x) and
            (character_neck_x < self._current_watch_top_right_x) and
            (character_neck_y < self._current_watch_top_left_y) and
            (character_neck_y > self._current_watch_bottom_left_y)
        ):

            return 0
        return -1

    def _get_approximated_character_position(self, obs) -> tuple[float, float]:
        """
        Returns v and u, where v is the number of vertical pixels from the top left corner of the image
        and u the number of horizontal pixels from the top left corner

        Illustration:

            u
        (0,0)→
         v ↓|-------------------------------|
            |                               |
            |       GAME IMAGE FRAME        |
            |                               |
            |-------------------------------|
        """

        matches = np.all(self._character_neck_rgb == obs, axis=-1)
        indices = np.argwhere(matches)

        if len(indices) > 0:
            v, u = indices[0]
            return v, u

        diff = obs - self._character_neck_rgb
        dist_squared = np.sum(diff ** 2, axis=-1)

        index_min = np.unravel_index(np.argmin(dist_squared), dist_squared.shape)

        v, u = index_min

        return v, u

    def _get_approximated_character_coords(self, obs):
        v, u = self._get_approximated_character_position(obs)
        # obs.shape =  (210, 160, 3)

        x = u
        y = game_image_frame_height - v

        return x, y

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping_reward = self._get_shaping_reward(obs)

        total_reward = (
            reward +
            shaping_reward
        )
        self.last_shaping = shaping_reward

        #  rgb(219 106 106)

        # Log detailed reward composition
        info['env_reward'] = reward
        info['shaping_reward'] = shaping_reward
        info['total_reward'] = total_reward

        if truncated:
            logger.debug('truncated!!!!!!!')

        if terminated:
            logger.debug('Terminated => checking for reaching guide pose again')
            self._stop_checking_for_rewards_util_next_terminate_state = False

        return obs, total_reward, terminated, truncated, info

    def _shaping(self, obs):
        # Example: encourage moving to the right
        return self.shaping_coef * obs[0]
