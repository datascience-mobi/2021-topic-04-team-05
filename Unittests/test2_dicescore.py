import unittest
import numpy as np
from sklearn.metrics import f1_score

from SVM_Segmentation.dicescore import dice_score


class TestDiceScore(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = np.asarray([[0, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0]])
        self.ground_truth = np.asarray([[0, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]])

    def test_dice_score(self):
        result = dice_score(self.prediction, self.ground_truth)
        result_by_hand = (2 * 12 / (2 * 12 + 2 + 2))  # 2*TP/2*TP+FP+FN
        self.assertEqual(result_by_hand, result)  # checks if a (1.0) == b (result)


# Comparing our dice with the python-integrated dice

prediction = np.asarray([[0, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0]])
ground_truth = np.asarray([[0, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]])

print(dice_score(prediction, ground_truth))
print(f1_score(ground_truth, prediction, average='micro'))
