import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import cv2
from SVM_Segmentation import readimages as rm
from SVM_Segmentation import preparation_PCA
import imagecodecs



from SVM_Segmentation import svm

class TestDiceScore(unittest.TestCase):
    def setUp(self) -> None:
        self.img = np.asarray(imread('29test.png'))

    def test_svm(self):
        result = svm.array_of_weights(self.img, self.img)
        result_SVC =
        self.assertEqual(result_SVC, result)  # checks if a (1.0) == b (result) (dice score should be 1 for image with itself)

        make_pipeline(StandardScaler(), SVC(class_weight='balanced', gamma=0.1))
        clf.fit(X, Y)