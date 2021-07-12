import unittest
import numpy as np
from matplotlib.image import imread

from SVM_Segmentation import dicescore

img = np.asarray([[0,1,1,0], [0,1,0,1],[1,0,0,0],[1,1,1,0]])
print(img)

gt = np.asarray([[0,1,0,0], [0,1,0,1],[1,0,0,1],[1,1,0,0]])
print(gt)


class TestDiceScore(unittest.TestCase):
    def setUp(self) -> None:
        self.img = np.asarray([[0,1,1,0], [0,1,0,1],[1,0,0,0],[1,1,1,0]])
        self.gt = np.asarray([[0,1,0,0], [0,1,0,1],[1,0,0,1],[1,1,0,0]])

    def test_dice_score(self):
        result = dicescore.dice_score(self.img, self.gt)
        #self.assertEqual(, result)  # checks if a (1.0) == b (result) (dice score should be 1 for image with itself)

