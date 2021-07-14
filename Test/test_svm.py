import numpy as np

#put SVM into testing_the_svm depending on what we use for testing (pictures: all SVM; test_dataframe X & y: part of SVM)

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([-1, 1, -1, -1])


class TestSVM(unittest.TestCase):
    def setUp(self) -> None:
        #self.img = np.asarray(imread('img29test.png'))
        #self.img2 = np.asarray(imread('gt29test.png'))
        self.X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        self.y = np.array([-1, 1, -1, -1])

    def test_svm(self):
        result = svm.main(self.X, self.y)
        model = svm.SVC(C=1, kernel='linear') #C = regularization strengh, kernel = linear?, gamma is not needed for linear kernel
        model.fit(X, y)
        result_SVC = model.predict([[-0.8, -1]])
        self.assertEqual(result_SVC, result)  # checks if a (1.0) == b (result)
