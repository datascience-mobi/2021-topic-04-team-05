import unittest
import numpy as np
from matplotlib.image import imread

from SVM_Segmentation.evaluation import dicescore


class TestDiceScore(unittest.TestCase):
    def setUp(self) -> None:
        self.img = np.asarray(imread('../Data/synthetic_cell_images/synthetic_masks/mask0.png'))

    def test_dice_score(self):
        result = dicescore.dice_score(self.img, self.img)
        self.assertEqual(1.0, result)  # checks if a (1.0) == b (result) (dice score should be 1 for image with itself)
