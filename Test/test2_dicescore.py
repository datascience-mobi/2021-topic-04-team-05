import unittest
import numpy as np

from SVM_Segmentation.dicescore import dice_score

class TestDiceScore(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = np.asarray([[0,1,1,0], [0,0,0,1],[1,0,0,0],[1,1,1,0]])
        self.groundtruth = np.asarray([[0,1,0,0], [0,1,0,1],[1,0,0,1],[1,1,0,0]])

    def test_dice_score(self):
        result = dice_score(self.prediction, self.groundtruth)
        result_by_hand = (2*12/(2*12+2+2)) #2*TP/2*TP+FP+FN
        self.assertEqual(result_by_hand, result)  # checks if a (1.0) == b (result)

