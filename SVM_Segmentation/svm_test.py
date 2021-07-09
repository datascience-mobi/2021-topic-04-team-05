import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from skimage.transform import resize
from skimage import io
import sklearn.decomposition as skdecomp
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from skimage.feature import multiscale_basic_features, canny
from skimage.filters import threshold_otsu
import cv2


# >>PIXEL CONVERSION<< #

def oneD_array_to_twoD_array(oneDarray):
    twoDarray = np.stack(oneDarray, axis=0)
    a = int(math.sqrt(len(twoDarray)))
    twoDarray = twoDarray.reshape(a, a)
    return twoDarray

def pca(image, variance):
    image = StandardScaler().fit_transform(image)
    pca = skdecomp.PCA(variance)
    pca.fit(image)
    components = pca.transform(image)
    projected = pca.inverse_transform(components)
    if projected is not None:
        return projected

##############################


# >> MODEL TRAINING << #
def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = regularization_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


# I haven't tested it but this same function should work for
# vanilla and mini-batch gradient descent as well
def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])  # gives multidimensional array

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W

        else:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw/len(Y_batch)  # average
    return dw


def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights

########################

# set hyper-parameters and call init
regularization_strength = 10000
learning_rate = 0.000001


if __name__ == '__main__':
    # read in images
    imgread = io.imread('../Data/test/img/t01.tif')
    imgresize = resize(imgread, (100, 100))
    imgcanny = canny(imgresize, sigma=.5)
    imgcannyflat = imgcanny.flatten()
    imgcannyflat = np.where(imgcannyflat == 'True', 1, imgcannyflat)
    thr = threshold_otsu(imgresize)
    imgotsu = (imgresize > thr).astype(float)
    imgotsuflat = imgotsu.flatten()
    imgflat = imgresize.flatten()
    imgnormal = np.hstack((imgflat, imgcannyflat, imgotsuflat))
    X = pd.DataFrame(data=imgnormal)
    X.insert(loc=len(X.columns), column='intercept', value=1)


    # read in ground truth images
    gtread = io.imread('../Data/test/gt/man_seg01.jpg')

    # thresholding ground truth images to get black-and-white-only images
    gtthreshold = (cv2.threshold(gtread, 0, 1, cv2.THRESH_BINARY))[1]
    gtresize = resize(gtthreshold, (100, 100))
    gtflat = gtthreshold.flatten()

    # Turning gt values into 1 and -1 labels
    y_labels = np.where(gtflat == 0, -1, gtflat)
    Y = pd.DataFrame(data=y_labels)

    # filter features
    #remove_correlated_features(X)
    #remove_less_significant_features(X, Y)

    # normalize data for better convergence and to prevent overflow
    # X_normalized = MinMaxScaler().fit_transform(X.values)
    # X = pd.DataFrame(X_normalized)

    # split data into train and test set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

    # train the model
    print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    # testing the model
    print("testing the model...")
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(X_train.to_numpy()[i], W))
        y_train_predicted = np.append(y_train_predicted, yp)

    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(X_test.to_numpy()[i], W))
        y_test_predicted = np.append(y_test_predicted, yp)

    print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))
    print("recall on test dataset: {}".format(recall_score(y_test, y_test_predicted)))
    print("precision on test dataset: {}".format(recall_score(y_test, y_test_predicted)))

    y_svm = np.array([])
    y_svm_train = np.append(y_svm, y_train_predicted)
    y_svm_test = np.append(y_svm_train, y_test_predicted)
    segmented_image = oneD_array_to_twoD_array(y_svm_test)
    plt.imshow(segmented_image)
    plt.show()