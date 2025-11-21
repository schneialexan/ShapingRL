from abc import ABC, abstractmethod


class RewardDefinition(ABC):

    def __init__(self, truncate_episode_after_n_steps: int):
        self._init_rewards(truncate_episode_after_n_steps=truncate_episode_after_n_steps)

    @abstractmethod
    def _init_rewards(self, truncate_episode_after_n_steps: int):
        raise NotImplementedError()

    def get_reward_definition_log_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class MontezumasTestShaperRewardDefinition(RewardDefinition):
    def _init_rewards(self, truncate_episode_after_n_steps: int):
        self._explored_all_grid_items_reward = 30
        self._explored_new_grid_item_reward = 1

        self.fall_reward = 0
        self.die_reward = -3

        self.explored_new_grid_item_reward = 1

        self.visit_already_explored_grid_item_reward = - 16 / truncate_episode_after_n_steps
        # self.move_in_x_axis_reward = 8 / truncate_episode_after_n_steps

        # self.visit_already_explored_grid_item_reward = 0
        self.move_in_x_axis_reward = 0
