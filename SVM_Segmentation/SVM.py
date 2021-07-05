import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import cv2
import readimages as rm
import PCA

#functions need for the loss function
def distance_of_point_to_hyperplane(w, x, y):
    return 1 - y * (np.dot(x, w))

def loss_function (x,w,y, C: float = 1e5):
    """
    This function calculates the loss of the support vectors.
    :param x: A dataframe with the features of the samples.
    :param w: The vector of the feature weights.
    :param y: A dataframe with the labels of the samples.
    :param C: A default value to define the regularization strength.
    :return: A value representing the loss.
    """
    #calculate hinge loss
    N = x.shape[0]
    separation = distance_of_point_to_hyperplane(w, x, y)
    separation = [0 if i < 0 else i for i in separation]
    hinge_loss = C * (np.sum(separation) / N)

    # calculate loss
    loss = 1 / 2 * np.dot(w, w) + hinge_loss
    return loss

#functions needed for the gradient
def distance_of_point_to_sv(index, w, x, y, C: float = 1e5):
    return w - (C * y[index] * x[index])

#calculating the gradient
def lagrange (x: np.array,w,y):
    """
    This function calculates the gradient of loss, which is then to be minimized.
    :param x: An array with the features of the samples.
    :param w: The vector of the feature weights.
    :param y: A dataframe with the labels of the samples.
    :return: A vector representing the gradient of the loss. #vector?
    """
    separation = distance_of_point_to_hyperplane(w, x, y)
    gradient = 0
    for index, q in enumerate(separation):
        # for correctly classified
        if q < 0:
            qi = w
        # for wrongly classified points
        else:
            qi = distance_of_point_to_sv(index, w, x, y)
        gradient += qi
    # calculate average of distances
    gradient = gradient/len(y)
    return gradient

#minimize gradient using Stochastic Gradient Descent
def stochastic_gradient_descent(features, labels, learning_rate: float = 1e-6):
    """
    This minimizes the gradient of loss, to find the global cost minimum.
    :param features: An array with the features of the samples.
    :param labels: A dataframe with the labels of the samples.
    :param learning_rate: A default value to define the learning rate, meaning the step size while performing the minimization of the gradient, of the SVM. (in percent)
    :return: The vector of the feature weights.
    """
    maximum_epochs = 5000
    weights = np.zeros(features.shape[1])
    power = 0
    unbounded_upper_value = float("inf")
    stoppage_criterion = 0.01  #in percent
    for epoch in range(1, maximum_epochs):
        # shuffel prevents the same x & y being take for several rounds
        x, y = shuffle(features, labels)
        for index, x in enumerate(x):
            upward_slope = lagrange(weights, x, y[index])
            weights = weights - (learning_rate * upward_slope)
        if epoch == pow(2, power) or epoch == maximum_epochs - 1:
            loss = loss_function(weights, features, labels)
            print("{}. epoch: current loss is {}.".format(epoch, loss))
            # stoppage criterion to stop at convergence
            deviance = abs(unbounded_upper_value - loss)
            # if cost no longer changes, stop gradient decend
            if stoppage_criterion * unbounded_upper_value > deviance:
                return weights
            unbounded_upper_value = loss
            power += 1
    return weights


def main(img_path, gt_path):
    """
    This minimizes the gradient of loss, to find the global cost minimum.
    :param img_path: The path of the images.
    :param gt_path: The path of the grount truth images.
    :return: The vector of the feature weights.
    """

    # read in microscopic images
    imageread = rm.read_image(img_path)
    image_PCA = PCA.convert_pca(imageread, 0.75)
    # normalizing microscopic images
    normalizedimg = []
    for i in range(0, len(imageread)):
        pixelsimg = imageread[i].astype('float32')
        if pixelsimg.max() > 0:
            normalimg = pixelsimg / pixelsimg.max()
            normalizedimg.append(normalimg)
        else:
            normalizedimg.append(pixelsimg)
    imagenames = rm.read_imagename(img_path)
    imageflattended = rm.image_flatten(image_PCA)

    X = rm.dataframe(imageflattended, imagenames)
    X.insert(loc=len(X.columns), column='intercept', value=1)

    # read in ground truth images
    gtread = rm.read_image(gt_path)

    # thresholding ground truth images to get black-and-white-only images
    thresholded = []
    for j in range(0, len(gtread)):
        threshold = cv2.threshold(gtread[j], 150, 255, cv2.THRESH_BINARY)
        thresholded.append(threshold[1])

    # normalizing ground truth images
    normalizedgt = []
    for k in range(0, len(thresholded)):
        pixelsgt = thresholded[k].astype('float32')
        if pixelsgt.max() > 0:
            normalgt = pixelsgt / pixelsgt.max()
            normalizedgt.append(normalgt)
        else:
            normalizedgt.append(pixelsgt)
    gtnames = rm.read_imagename(gt_path)
    thresholded_and_normalized_flattened = rm.image_flatten(normalizedgt)

    y = rm.dataframe(thresholded_and_normalized_flattened, gtnames)  # ground truths

    # Cross validation to train the model with different train:test splits
        # leave-one-out cross-validation: n_splits = number of samples
    n_splits = 2
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=None)

    # define test and training data
    for i in range(n_splits):
        result = next(kfold.split(X), None)
        X_train = X.iloc[result[0]]
        # !!X_train = np.array([X.iloc[result[0]]]) statt unten .to_numpy()
        X_test = X.iloc[result[1]]
        y_train = y.iloc[result[0]]
        y_test = y.iloc[result[1]]

        # train the model
        W = stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy())
        print("The weights vector is: {}".format(W))

    # use model to predict y for the training data
    y_train_prediction = np.array([])
    for i in range(X_train.shape[0]):
        # sign returns -1 if x < 0, 0 if x==0, 1 if x > 0
        y_pred = np.sign(np.dot(X_train.to_numpy()[i], W))
        y_train_prediction = np.append(y_train_prediction, y_pred)

    # test model
    y_test_prediction = np.array([])
    for i in range(X_test.shape[0]):
        # sign returns -1 if x < 0, 0 if x==0, 1 if x > 0
        y_pred = np.sign(np.dot(X_test.to_numpy()[i], W))
        y_test_prediction = np.append(y_test_prediction, y_pred)

    return y_test_prediction, y_train_prediction

if __name__ == '__main__':
    main('../Data/test/img', '../Data/test/gt')


