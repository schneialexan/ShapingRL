import logging
from collections import deque
from copy import copy
from typing import Self, List

import gymnasium as gym
import numpy as np

from info.montezumas_info import MontezumaActions
from information_extraction.montezumas_revenge import MontezumasInformationExtractor
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

        guideline: Guideline = (
            guide_builder.set_goal_go_to(
                name="Right ladder top",
                grid=8,
                cell=5,
                row=4,
                n_times=100
            )
            .then_set_goal_go_to(
                name="Right ladder middle",
                grid=8,
                row=3,
                cell=5,
                n_times=100
            )
            .then_set_goal_go_to(
                name="Right ladder bottom",
                grid=8,
                row=2,
                cell=5,
                n_times=100
            )
            .then_set_goal_go_to(
                name="Right ladder bottom with little offset left",
                grid=8,
                row=3,
                cell=3,
                n_times=100
            )
            .then_set_goal_go_to(
                name="Left ladder bottom",
                grid=8,
                row=2,
                cell=1,
                n_times=100
            ).build())

        self._guideline_coach: IGuidelineCoach = guideline_coach(
            guideline
        )

        self._last_step_number_of_lives = None

        self._action_history = deque()

    def _get_shaping_reward(self, obs, action, agent_lost_live: bool, agent_terminated: bool,
                            agent_had_n_lives_when_lost_life: int | None) -> float:
        shape_reward = 0

        if agent_lost_live:
            penality = (1 / agent_had_n_lives_when_lost_life) * (-50)
            logger.info(f'Agent lost live, penalise (burn to hell!!!!!), {penality = }')
            shape_reward += penality
            self._action_history = deque()

        shape_reward += self._guideline_coach.get_reward(obs)

        if agent_terminated:
            logger.info('Agent died funeral rip :-(')
            self._last_step_number_of_lives = None
            self._guideline_coach.on_episode_termination()
            shape_reward += (-100)
            self._action_history = deque()

        if (
            (action == MontezumaActions.DOWN.value) or
            (action == MontezumaActions.DOWNRIGHT.value) or
            (action == MontezumaActions.DOWNLEFT.value)
        ) and (
            (MontezumaActions.UP.value in self._action_history) or
            (MontezumaActions.UPRIGHT.value in self._action_history) or
            (MontezumaActions.UPLEFT.value in self._action_history)
        ):
            #logger.info('up down loop detected -3')
            shape_reward += (-1_000)

        if (
            (action == MontezumaActions.UP.value) or
            (action == MontezumaActions.UPLEFT.value) or
            (action == MontezumaActions.UPRIGHT.value)
        ) and (
            (MontezumaActions.DOWN.value in self._action_history) or
            (MontezumaActions.DOWNLEFT.value in self._action_history) or
            (MontezumaActions.DOWNRIGHT.value in self._action_history)
        ):
            #logger.info('up down loop detected -3')
            shape_reward += (-1_000)

        # if action in [
        #     MontezumaActions.UP.value,
        #     MontezumaActions.DOWN.value,
        #     MontezumaActions.UPLEFT.value,
        #     MontezumaActions.DOWNLEFT.value,
        #     MontezumaActions.UPRIGHT.value,
        #     MontezumaActions.DOWNRIGHT.value
        # ]:
        if len(self._action_history) == 7:
            self._action_history.popleft()

        self._action_history.append(action)
        # logger.info(f'{shape_reward = }')

        return shape_reward

    def step(self, action):
        agent_had_n_lives_when_lost_life = None
        obs, reward, terminated, truncated, info = self.env.step(action)

        # logger.info(f'agent action = {MontezumaActions(action)}')

        current_number_of_lives = MontezumasInformationExtractor.get_number_of_lives(obs)

        agent_lost_live = False
        if (
            (self._last_step_number_of_lives is not None) and
            (self._last_step_number_of_lives > current_number_of_lives)
        ):
            agent_lost_live = True

            logger.info(
                f'Agent\'s number of lives changed {self._last_step_number_of_lives} => {current_number_of_lives}'
            )
            agent_had_n_lives_when_lost_life = self._last_step_number_of_lives
            self._last_step_number_of_lives = current_number_of_lives
        elif self._last_step_number_of_lives is None:
            self._last_step_number_of_lives = current_number_of_lives

        shaping_reward = self._get_shaping_reward(
            obs, agent_lost_live=agent_lost_live,
            action=action,
            agent_terminated=terminated,
            agent_had_n_lives_when_lost_life=agent_had_n_lives_when_lost_life
        )

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
            logger.info('truncated!!!!!!!')

        # logger.info(f'{total_reward= }')

        return obs, total_reward, terminated, truncated, info
