import unittest
import numpy as np
from matplotlib.image import imread

from dicescore import dice_score

img1 = np.asarray(imread('man_seg21testing.png'))

class TestDiceScore(unittest.TestCase):
    def test_dice_score(self):
        result = dice_score(img1, img1)
        self.assertEqual(result, 1.0) #checks if a (result) == b (1) (dice score should be 1 for image with itself)

