from abc import ABC, abstractmethod
from collections import deque
import logging

import numpy as np

from info.montezumas_info import GAME_IMAGE_FRAME_HEIGHT, GAME_IMAGE_FRAME_WIDTH
from log.montezumas import configure_logging_for_montezumas
from training.guide import Guideline, GuidelineItem

configure_logging_for_montezumas('coach')
logger = logging.getLogger(__name__)


class IGuidelineCoach(ABC):
    @abstractmethod
    def __init__(self, guideline: Guideline):
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self, observations) -> float:
        raise NotImplementedError()

    def on_episode_termination(self):
        pass


class MontezumasKeyGuidelineCoach(IGuidelineCoach):

    def __init__(self, guideline: Guideline):
        self._guideline: Guideline = guideline

        self._character_neck_rgb = np.array([228, 111, 111])

        self._reach_guide_point_reward: float = 100

        self._guideline_items: deque[GuidelineItem] = self._guideline.get_guide_items()

        self._current_guide_item: GuidelineItem | None = None

        self._current_watch_top_left_x = None
        self._current_watch_top_left_y = None

        self._current_watch_bottom_left_x = None
        self._current_watch_bottom_left_y = None

        self._current_watch_top_right_x = None
        self._current_watch_top_right_y = None

        self._current_watch_bottom_right_x = None
        self._current_watch_bottom_right_y = None
        self._stop_checking_for_rewards_util_next_terminate_state = False

    def get_reward(self, observations) -> float:

        if self._stop_checking_for_rewards_util_next_terminate_state:
            return 0

        if (self._current_guide_item is None) or self._current_guide_item.is_completed:
            self._current_guide_item = self._guideline_items.popleft()

            logger.info(f'using new guide point with name "{self._current_guide_item.name}"')

            grid_cell_width = GAME_IMAGE_FRAME_WIDTH / self._current_guide_item.grid
            grid_cell_height = GAME_IMAGE_FRAME_HEIGHT / self._current_guide_item.grid

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

        character_neck_x, character_neck_y = self._get_approximated_character_coords(observations)

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
        y = GAME_IMAGE_FRAME_HEIGHT - v

        return x, y

    def on_episode_termination(self):
        self._stop_checking_for_rewards_util_next_terminate_state = False


class MontezumasKeyGuidelineCoachForcingOrder(MontezumasKeyGuidelineCoach):
    def __init__(self, guideline: Guideline):
        self._reached_item_indexes = []
        super().__init__(guideline)

        self._max_reached_guide_item_index = -1

    def get_reward(self, observations) -> float:
        total_shape_rewards = 0
        for index, guideline_item in enumerate(self._guideline_items):
            total_shape_rewards += self._process(
                guide_line_item_index=index,
                guideline_item=guideline_item,
                observations=observations
            )

        return total_shape_rewards

    def _process(self, guide_line_item_index: int, guideline_item: GuidelineItem, observations):

        if guide_line_item_index in self._reached_item_indexes:
            return 0

        grid_cell_width = GAME_IMAGE_FRAME_WIDTH / guideline_item.grid
        grid_cell_height = GAME_IMAGE_FRAME_HEIGHT / guideline_item.grid

        current_watch_top_left_y = guideline_item.row * grid_cell_height
        current_watch_bottom_left_y = (
                                          guideline_item.row - 1
                                      ) * grid_cell_height

        current_watch_top_right_y = current_watch_top_left_y
        current_watch_bottom_right_y = current_watch_bottom_left_y

        current_watch_top_left_x = (guideline_item.cell - 1) * grid_cell_width
        current_watch_top_right_x = guideline_item.cell * grid_cell_width

        current_watch_bottom_left_x = current_watch_top_left_x
        current_watch_bottom_right_x = current_watch_top_right_x

        character_neck_x, character_neck_y = self._get_approximated_character_coords(observations)

        if (
            (guide_line_item_index > self._max_reached_guide_item_index) and
            (character_neck_x > current_watch_top_left_x) and
            (character_neck_x < current_watch_top_right_x) and
            (character_neck_y < current_watch_top_left_y) and
            (character_neck_y > current_watch_bottom_left_y)
        ):
            guideline_item.visit()

            reward = 2 ** (guide_line_item_index + 1)

            logger.info(
                f'Reached goal pos for "{guideline_item.name}" ({guideline_item.n_times_visited}/{guideline_item.n_times}) => give reward {reward}'
            )

            # del self._guideline_items[guide_line_item_index]
            self._reached_item_indexes.append(guide_line_item_index)
            self._max_reached_guide_item_index = guide_line_item_index

            return reward  # has to be smaller than first game reward todo: ensure that!!!
        return 0

    def on_episode_termination(self):
        # logger.info(
        #     'Episode terminated, rewards can now be received again from all guideline items')
        self._guideline_items = self._guideline.get_guide_items()
        self._max_reached_guide_item_index = -1
        self._reached_item_indexes.clear()
