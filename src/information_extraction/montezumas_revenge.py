from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import gymnasium as gym


class MontezumasRAMInformationExtractor:
    def __init__(self, wrapper: gym.Wrapper):
        self._wrapper: gym.Wrapper = wrapper

        # copied from https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
        self._variable_to_ram_index_dict = dict(
            room_number=3,
            player_x=42,
            player_y=43,
            player_direction=52,
            # 72:  facing left, 40:  facing left, climbing down ladder/rope 24:  facing left, climbing up ladder/rope 128: facing right 32:  facing right, climbing down ladder/rope, 16: facing right climbing up ladder/rope
            enemy_skull_x=47,
            enemy_skull_y=46,
            key_monster_x=44,
            key_monster_y=45,
            level=57,
            num_lives=58,
            items_in_inventory_count=61,
            room_state=62,
            score_0=19,
            score_1=20,
            score_2=21
        )

    def get_ram(self):
        return self._wrapper.env.unwrapped.ale.getRAM()

    def get_player_position(self):
        """Returns x and y position of the player

        Returns tuple:
            0: x
            1: y
        """
        ram = self._wrapper.env.unwrapped.ale.getRAM()
        return int(ram[
                       self._variable_to_ram_index_dict["player_x"]
                   ]), int(ram[
                               self._variable_to_ram_index_dict["player_y"]
                           ]) - 107

    def get_player_number_of_lives(self):
        ram = self._wrapper.env.unwrapped.ale.getRAM()
        return ram[
            self._variable_to_ram_index_dict["num_lives"]
        ]

    def get_room_id(self):
        ram = self._wrapper.env.unwrapped.ale.getRAM()
        return ram[
            self._variable_to_ram_index_dict["room_state"]
        ]


class MontezumasInformationExtractor(ABC):
    GAME_IMAGE_FRAME_HEIGHT = 210
    GAME_IMAGE_FRAME_WIDTH = 160

    CHARACTER_NECK_RGB = np.array([228, 111, 111])

    @staticmethod
    def draw_rect(x, y, width, height, ax, label_top=None, edgecolor='red', fill_color=None):
        # Rectangle: (x, y) is top-left corner

        additional_params_rect = {}
        if fill_color is not None:
            additional_params_rect["facecolor"] = fill_color
        else:
            additional_params_rect["facecolor"] = 'none'  # transparent fill

        if edgecolor is not None:
            additional_params_rect['edgecolor'] = edgecolor
        else:
            additional_params_rect['edgecolor'] = 'none'

        rect = patches.Rectangle(
            (x, MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT - y),  # (x, y) position
            width, height,  # width, height
            linewidth=1,
            **additional_params_rect
        )

        ax.text(
            x + 1, MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT - y - 3, label_top,
            color='lime',
            fontsize=6,
        )

        ax.add_patch(rect)

        return ax

    @staticmethod
    def get_number_of_lives(obs: np.array):
        # Display it using matplotlib

        rect_size = 5

        base_x = (MontezumasInformationExtractor.GAME_IMAGE_FRAME_WIDTH // 2) - (rect_size / 2)
        y = MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT - 14

        positions = [
            (base_x - 21, y),
            (base_x - 13, y),
            (base_x - 5, y),
            (base_x + 3, y),
            (base_x + 11, y),
        ]

        # fig, ax = plt.subplots()
        # ax.imshow(obs)
        # ax.axis('off')  # hide axes
        # fig.tight_layout()
        # for index, (x, y) in enumerate(positions):
        #     MontezumasInformationExtractor.draw_rect(
        #         x,
        #         y,
        #         rect_size,
        #         rect_size,
        #         ax,
        #         label_top=str(index + 1)
        #     )
        # plt.show()

        for index, (x, y) in reversed(list(enumerate(positions))):
            x = int(x)
            y = int(MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT - y)

            roi = obs[y:y + rect_size, x:x + rect_size]
            has_non_black = np.any(roi != 0)

            if has_non_black:
                return index + 1

        return 0

    @staticmethod
    def get_approximated_character_position(obs) -> tuple[float, float]:
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

        matches = np.all(MontezumasInformationExtractor.CHARACTER_NECK_RGB == obs, axis=-1)
        indices = np.argwhere(matches)

        if len(indices) > 0:
            v, u = indices[0]
            return v, u

        diff = obs - MontezumasInformationExtractor.CHARACTER_NECK_RGB
        dist_squared = np.sum(diff ** 2, axis=-1)

        index_min = np.unravel_index(np.argmin(dist_squared), dist_squared.shape)

        v, u = index_min

        return v, u

    @staticmethod
    def get_approximated_character_coords(obs):
        v, u = MontezumasInformationExtractor.get_approximated_character_position(obs, )
        # obs.shape =  (210, 160, 3)

        x = u
        y = MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT - v

        return x, y
