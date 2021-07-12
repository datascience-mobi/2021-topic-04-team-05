
import unittest
import numpy as np
from matplotlib.image import imread

from SVM_Segmentation.Synthetic_images.syntheticmask import syntheticmask

class Testsyntheticmask(unittest.TestCase):
    def setUp(self) -> None:
        self.img = np.asarray(imread('gt29test.png'))

    def test_syntheticmask(self):
