import logging
from collections import deque
from copy import copy
from typing import Self, List

import gymnasium as gym
import numpy as np

from log.montezumas import configure_logging_for_montezumas
from training.coach import MontezumasKeyGuidelineCoach, IGuidelineCoach
from training.guide import GuideLineBuilder, Guideline

configure_logging_for_montezumas('guide_shaper')
logger = logging.getLogger(__name__)


class NaiveDenseWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        guideline_coach: type[IGuidelineCoach],
        shaping_coef=0.1
    ):
        super().__init__(env)
        self.shaping_coef = shaping_coef
        self.last_shaping = 0.0

        guide_builder = GuideLineBuilder()

        guideline: Guideline = guide_builder.set_goal_go_to(
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
        ).build()

        self._guideline_coach: IGuidelineCoach = guideline_coach(
            guideline
        )

    def _get_shaping_reward(self, obs) -> float:
        return self._guideline_coach.get_reward(obs)

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
            self._guideline_coach.on_episode_termination()

        return obs, total_reward, terminated, truncated, info

    def _shaping(self, obs):
        # Example: encourage moving to the right
        return self.shaping_coef * obs[0]
