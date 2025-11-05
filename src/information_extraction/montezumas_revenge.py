from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


class MontezumasInformationExtractor(ABC):
    GAME_IMAGE_FRAME_HEIGHT = 210
    GAME_IMAGE_FRAME_WIDTH = 160

    CHARACTER_NECK_RGB = np.array([228, 111, 111])

    @staticmethod
    def draw_rect(x, y, width, height, ax, label_top=None, edgecolor='red', fill_color=None):
        # Rectangle: (x, y) is bottom-left corner

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
            # backgroundcolor='black'
        )

        # Add the rectangle to the Axes
        ax.add_patch(rect)

        return ax

    @staticmethod
    def get_health(obs: np.array):
        # Display it using matplotlib
        fig, ax = plt.subplots()
        ax.imshow(obs)
        ax.axis('off')  # hide axes
        fig.tight_layout()

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

    def get_approximated_character_position(self, obs) -> tuple[float, float]:
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

    def get_approximated_character_coords(self, obs):
        v, u = MontezumasInformationExtractor.get_approximated_character_position(obs)
        # obs.shape =  (210, 160, 3)

        x = u
        y = MontezumasInformationExtractor.GAME_IMAGE_FRAME_HEIGHT - v

        return x, y
