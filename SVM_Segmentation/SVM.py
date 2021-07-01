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
    :return: A value representing the gradient of the loss.
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
    This function calculates the gradient of loss, which is then to be minimized.
    :param x: An array with the features of the samples.
    :param w: The vector of the feature weights.
    :param y: A dataframe with the labels of the samples.
    :return: A value representing the loss.
    """
    maximum_epochs = 5000 #an epoch indicates the number of passes of the entire training dataset the machine learning algorithm has completed
    weights = np.zeros(features.shape[1])  #creating array filled with zeros of the number of columns of our features (d.h. so viele wie features) dataset
    power = 0 #hoch
    unbounded_upper_value = float("inf") #acts as unbounded upper value for comparison for finding lowest values of something
    stoppage_criterion = 0.01  #in percent
    # stochastic gradient descent
    for epoch in range(1, maximum_epochs):
        x, y = shuffle(features, labels)  # shuffle to prevent repeating update cycles; Stichproben von
        # features & outputs werden rausgezogen (jede Runde neu)
        # Stichproben von features & outputs werden rausgezogen (jede Runde neu)
        for index, x in enumerate(x):
            upward_slope = lagrange(weights, x, y[index]) #ascend = average distance
            weights = weights - (learning_rate * upward_slope) #move opposite to the gradient by a certain rate (s. Diagramm J(w) zu w; learning_rate = Schrittgröße in Prozent --> Schrittgröße wird kleiner mit sinkender Steigung)
        if epoch == pow(2, power) or epoch == maximum_epochs - 1: #2 hoch iwas oder 4999
            loss = loss_function(weights, features, labels) #calculate cost, wird immer kleiner
            print("{}. epoch: current loss is {}.".format(epoch, loss))
            # stoppage criterion
            deviance = abs(unbounded_upper_value - loss)
            if stoppage_criterion * unbounded_upper_value > deviance: #wenn bedingung erfüllt wird loop gestoppt; prev_cost - cost wird immer kleiner & wird irgendwann kleiner als 0.01% des prev_cost (0.01% des prev_cost ist die von uns definierte threshold) -> stoppt also wenn keine Änderung der cost mehr passiert
                return weights #output of very last iteration
            unbounded_upper_value = loss #prev_cost ist erst infinite & wird dann immer kleiner, da cost kleiner wird
            power += 1 #iteration
    return weights #für for loop (wird dafür gebraucht)


def main(img_path, gt_path):
    # read dataset
    # X = data.features
    # y = data.labels

    # read in normal images
    # rm = readimages.py
    imageread = rm.read_image(img_path)  # Bilder eines Ordners in Liste mit 2D arrays
    image_PCA = PCA.convert_pca(imageread, 0.75)
    normalizedimg = []
    for i in range(0, len(imageread)):
        pixelsimg = imageread[i].astype('float32')
        if pixelsimg.max() > 0:
            normalimg = pixelsimg / pixelsimg.max()
            normalizedimg.append(normalimg)
        else:
            normalizedimg.append(pixelsimg)
    imagenames = rm.read_imagename(img_path)  # Liste mit Namen der Bilder
    imageflattended = rm.image_flatten(image_PCA)

    X = rm.dataframe(imageflattended, imagenames)
    X.insert(loc=len(X.columns), column='intercept', value=1)

    # read in gt images
    gtread = rm.read_image(gt_path)  # Bilder eines Ordners in Liste mit 2D arrays

    # thresholding gt images
    thresholded = []
    for j in range(0, len(gtread)):
        threshold = cv2.threshold(gtread[j], 150, 255, cv2.THRESH_BINARY)  # 0-149, intensitätswert wird 0, 150-255 intensitätswert wird 1
        thresholded.append(threshold[1])

    #normalizing gt images
    normalizedgt = []
    for k in range(0, len(thresholded)):
        pixelsgt = thresholded[k].astype('float32')
        if pixelsgt.max() > 0:
            normalgt = pixelsgt / pixelsgt.max()
            normalizedgt.append(normalgt)
        else:
            normalizedgt.append(pixelsgt)
    gtnames = rm.read_imagename(gt_path)  # Liste mit Namen der Bilder
    thresholded_and_normalized_flattened = rm.image_flatten(normalizedgt)
    y = rm.dataframe(thresholded_and_normalized_flattened, gtnames)  # ground truths

    # Cross validation to train the model with different train:test splits
    # leave-one-out cross-validation: n_splits = number of samples
    n_splits = 2
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=None)

    for i in range(n_splits):
        # next creates an iterator, and prints the items one by one
        result = next(kfold.split(X), None)
        X_train = X.iloc[result[0]]
        # !!X_train = np.array([X.iloc[result[0]]]) statt unten .to_numpy()
        X_test = X.iloc[result[1]]
        y_train = y.iloc[result[0]]
        y_test = y.iloc[result[1]]
        # train the model
        W = stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy())
        print("The weights vector is: {}".format(W))

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


if __name__ == '__main__':
    main('../Data/N2DH-GOWT1/img', '../Data/N2DH-GOWT1/gt/tif')


