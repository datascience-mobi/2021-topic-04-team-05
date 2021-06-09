import unittest
import numpy as np
from matplotlib.image import imread

from functions.dicescore_code import dicescore


class TestDiceScore(unittest.TestCase):
    def setUp(self) -> None:
        self.img = np.asarray(imread('29test.png'))

    def test_dice_score(self):
        result = dicescore.dice_score(self.img, self.img)
        self.assertEqual(result, 1.0)  # checks if a (result) == b (1) (dice score should be 1 for image with itself)
