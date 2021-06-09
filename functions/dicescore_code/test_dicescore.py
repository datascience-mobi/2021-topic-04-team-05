import unittest
import numpy as np
from matplotlib.image import imread

from functions.dicescore_code import dicescore


class TestDiceScore(unittest.TestCase):
    def setUp(self) -> None:
        self.img = np.asarray(imread('29test.png'))

    def test_dice_score(self):
        result = dicescore.dice_score(self.img, self.img)
        self.assertEqual(1.0, result)  # checks if a (1.0) == b (result) (dice score should be 1 for image with itself)

#nicholas code

import unittest
import numpy as np
from functions.dicescore_code import dicescore


class TestDiceScore(unittest.TestCase):
    def setUp(self) -> None:
        #     self.img = np.asarray(imread(pl.Path("../29test.png'))
        self.h_img = np.array([[0, 0, 0],
                               [1, 1, 1],
                               [0, 0, 0]])

        self.v_img = np.array([[0, 1, 0],
                               [0, 1, 0],
                               [0, 1, 0]])

    def test_dice_score_identity(self):
        result = dicescore.dice_score(self.h_img, self.h_img)
        self.assertEqual(1.0, result)  # checks if a (result) == b (1) (dice score should be 1 for image with itself)

    def test_dice_score_hv(self):
        result = dicescore.dice_score(self.h_img, self.v_img)

        self.assertEqual(0.75, result)