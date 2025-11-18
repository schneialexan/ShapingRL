import os.path
import unittest

import numpy as np
from PIL import Image

from information_extraction.montezumas_revenge import MontezumasInformationExtractor
from util import PROJECT_ROOT_PATH


class TestMontezumasInformationExtractorTest(unittest.TestCase):
    def test_get_health_is_correct_when_game_shows_5_lives(self):
        img = np.array(
            Image.open(
                os.path.join(
                    PROJECT_ROOT_PATH,
                    "tests",
                    "information_extraction",
                    "frames",
                    "montezumas",
                    "5_lives.png",
                )
            )
        )
        n_lives = MontezumasInformationExtractor.get_number_of_lives(img)

        self.assertEqual(5, n_lives)

    def test_get_health_is_correct_when_game_shows_4_lives(self):
        img = np.array(
            Image.open(
                os.path.join(
                    PROJECT_ROOT_PATH,
                    "tests",
                    "information_extraction",
                    "frames",
                    "montezumas",
                    "4_lives.png",
                )
            )
        )
        n_lives = MontezumasInformationExtractor.get_number_of_lives(img)

        self.assertEqual(4, n_lives)

    def test_get_health_is_correct_when_game_shows_3_lives(self):
        img = np.array(
            Image.open(
                os.path.join(
                    PROJECT_ROOT_PATH,
                    "tests",
                    "information_extraction",
                    "frames",
                    "montezumas",
                    "3_lives.png",
                )
            )
        )
        n_lives = MontezumasInformationExtractor.get_number_of_lives(img)

        self.assertEqual(3, n_lives)

    def test_get_health_is_correct_when_game_shows_2_lives(self):
        img = np.array(
            Image.open(
                os.path.join(
                    PROJECT_ROOT_PATH,
                    "tests",
                    "information_extraction",
                    "frames",
                    "montezumas",
                    "2_lives.png",
                )
            )
        )
        n_lives = MontezumasInformationExtractor.get_number_of_lives(img)

        self.assertEqual(2, n_lives)

    def test_get_health_is_correct_when_game_shows_1_live(self):
        img = np.array(
            Image.open(
                os.path.join(
                    PROJECT_ROOT_PATH,
                    "tests",
                    "information_extraction",
                    "frames",
                    "montezumas",
                    "1_lives.png",
                )
            )
        )
        n_lives = MontezumasInformationExtractor.get_number_of_lives(img)

        self.assertEqual(1, n_lives)

    def test_get_health_is_correct_when_game_shows_0_live(self):
        img = np.array(
            Image.open(
                os.path.join(
                    PROJECT_ROOT_PATH,
                    "tests",
                    "information_extraction",
                    "frames",
                    "montezumas",
                    "0_lives.png",
                )
            )
        )
        n_lives = MontezumasInformationExtractor.get_number_of_lives(img)

        self.assertEqual(0, n_lives)
