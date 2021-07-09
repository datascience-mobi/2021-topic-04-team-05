import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import cv2
import readimages as rm
import preparation_PCA
from SVM_Segmentation import preparation_tiles as pt


# functions need for the loss function
def distance_of_point_to_hyperplane(w, x, y):
    """

    :param w:
    :param x:
    :param y:
    :return:
    """
    if x.shape == (1,):
        x = x[0]
    if w.shape == (1,):
        w = w[0]
    distance_hyperplane = 1 - y * (np.dot(x, w))
    return distance_hyperplane


def loss_function(w, x, y, soft_margin_parameter: float = 1e5):
    """
    This function calculates the loss of the support vectors.
    :param x: A dataframe with the features of the samples.
    :param w: The vector of the feature weights.
    :param y: A dataframe with the labels of the samples.
    :param C: A default value to define the regularization strength.
    :return: A value representing the loss.
    """
    # calculate hinge loss
    N = x.shape[0]
    separation = distance_of_point_to_hyperplane(w, x, y)
    separation = separation[0, :]
    separation = [0 if i < 0 else i for i in separation]
    hinge_loss = soft_margin_parameter * (np.sum(separation) / N)

    # calculate loss
    loss = 1 / 2 * np.dot(w, w) + hinge_loss
    return loss


# functions needed for the gradient
def distance_of_point_to_sv(weights, features, labels, soft_margin_parameter: float = 1e5):
    """

    :param index:
    :param w:
    :param x:
    :param y:
    :param C:
    :return:
    """
    distance_sv = weights - (soft_margin_parameter * labels * features)
    return distance_sv


# calculating the gradient
def lagrange(weights, features, labels, distances_to_hyperplane: list):
    """
    This function calculates the gradient of loss, which is then to be minimized.
    :param weights: an array of weights, with number of columns equivalent to number of pixels of a picture
    :param features: one column/feature of a dataframe of features
    :param labels: one column of a dataframe of labels
    :param distances_to_hyperplane: list of arrays, with the length of number of pixels, and dimension of arrays
    is equivalent to number of features
    :return: gradient for one image/feature
    """

    if type(features) == np.float64:
        labels = np.array([labels])
        features = np.array([features])

    gradient = 0
    # iterating trough all rows (for every pixel)
    for index1, distance_to_hyperplane in enumerate(distances_to_hyperplane):
        # iterating through all values of the different features
        # for correctly classified
        if distance_to_hyperplane < 0:
            distances_to_sv = weights
        # for falsely classified points
        else:
            # calculating the distance to the support vector for every feature
            for index2 in range(0, len(list(features))):
                distances_to_sv = distance_of_point_to_sv(weights, features[index2], labels)

        gradient += distances_to_sv
        # calculate average of distances
    gradient = gradient / len([labels])
    return gradient


# minimize gradient using Stochastic Gradient Descent
def stochastic_gradient_descent(features, labels, learning_rate: float = 1e-6):
    """
    This minimizes the gradient of loss, to find the global cost minimum.
    :param features: An array with the features of the samples.
    :param labels: A dataframe with the labels of the samples.
    :param learning_rate: A default value to define the learning rate, meaning the step size while performing the minimization of the gradient, of the SVM. (in percent)
    :return: The vector of the feature weights.
    """
    maximum_epochs = 5000
    power = 0
    unbounded_upper_value = float("inf")
    stoppage_criterion = 0.01  # in percent
    for epoch in range(1, maximum_epochs):
        # shuffle prevents the same x & y being taken for several rounds
        x, y = shuffle(features, labels)
        x = pd.DataFrame.to_numpy(x)
        x = x.transpose()
        y = pd.DataFrame.to_numpy(y)
        y = y.transpose()
        i = 0
        for j in range(0, y.shape[1] - 1):
            y = y[:, [j]]
            if number_of_features != 0:
                for i in range(0, x.shape[1] - 1):
                    end = i + number_of_features
                    x = x[:, [i, end]]
                    i += number_of_features
            else:
                for i in range(0, x.shape[1] - 1):
                    x = x[:, [i]]
            distances_to_hyperplane = []
            intercept = np.zeros((x.shape[0], 1), dtype=x.dtype)
            intercept += 1
            x_with_intercept = np.hstack((x, intercept))
            array_of_weights = np.zeros(x_with_intercept.shape[1])
            for index in range(0, x_with_intercept.shape[0]):
                # distance is always a value, also for multiple features
                distance_to_hyperplane = distance_of_point_to_hyperplane(array_of_weights, x_with_intercept[index],
                                                                         y[index])
                # creating a list with all of the distances, for each pixel
                distances_to_hyperplane.append(distance_to_hyperplane)
                # we calculate the gradient for one picture/column
                gradient = lagrange(array_of_weights, x_with_intercept[index], y[index], distances_to_hyperplane)
                array_of_weights = array_of_weights - (learning_rate * gradient)
            if epoch == pow(2, power) or epoch == maximum_epochs - 1:
                # calculate the loss
                # array_of_weights = np.asarray(array_of_weights)
                loss = loss_function(array_of_weights, x_with_intercept, y)
                print("{}. epoch: current loss is {}.".format(epoch, loss))
                # stoppage criterion to stop at convergence
                deviance = abs(unbounded_upper_value - loss)
                # if cost no longer changes, stop gradient decend
                if stoppage_criterion * unbounded_upper_value > deviance:
                    print(array_of_weights)
                unbounded_upper_value = loss
                power += 1
    print(array_of_weights)


def main(img_dataframe, gt_dataframe, number_of_features):
    """
    This minimizes the gradient of loss, to find the global cost minimum.
    :param img_path: The path of the images.
    :param gt_path: The path of the ground truth images.
    :return: The vector of the feature weights.
    """

    # normalizing microscopic images
    img_array = img_dataframe.values
    img_normalized_array = MMS().fit_transform(img_array)
    img_normalized_df = pd.DataFrame(img_normalized_array, columns=img_dataframe.columns)

    # thresholding ground truth images to get black-and-white-only images
    gt_threshold_array = cv2.threshold(gt_dataframe.values, 0, 1, cv2.THRESH_BINARY)
    gt_threshold_df = pd.DataFrame(gt_threshold_array[1], columns=gt_dataframe.columns)
    gt_labels_df = gt_threshold_df.replace(0, -1)

    # Cross validation to train the model with different train:test splits
    # leave-one-out cross-validation: n_splits = number of samples
    n_splits = 2
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=None)

    # define test and training data
    for i in range(n_splits):
        split_data = next(kfold.split(img_normalized_df.transpose()), None)
        gt_train = gt_labels_df.iloc[:, split_data[0]]
        gt_test = gt_labels_df.iloc[:, split_data[1]]

    if number_of_features != 0:
        for j in range(0, len(split_data)):
            for i in range(0, split_data[j].size):
                split_train = split_data[j]
                split_train_value = split_train[i]
                img_train = img_normalized_df.iloc[:, split_train_value: split_train_value+number_of_features]
                img_test = img_normalized_df.iloc[:, split_data[j]]

                # train the model
                W = stochastic_gradient_descent(img_train.to_numpy(), gt_train.to_numpy())
                print("The weights vector is: {}".format(W))

                # use model to predict y for the training data
                y_train_prediction = np.array([])
                for i in range(img_train.shape[0]):
                    # sign returns -1 if x < 0, 0 if x==0, 1 if x > 0
                    y_pred = np.sign(np.dot(img_train.to_numpy()[i], W))
                    y_train_prediction = np.append(y_train_prediction, y_pred)

                # test model
                y_test_prediction = np.array([])
                for i in range(img_test.shape[0]):
                    # sign returns -1 if x < 0, 0 if x==0, 1 if x > 0
                    y_pred = np.sign(np.dot(img_test.to_numpy()[i], W))
                    y_test_prediction = np.append(y_test_prediction, y_pred)

    return y_test_prediction, y_train_prediction


if __name__ == '__main__':
    imageread = pt.tiles('../Data/N2DH-GOWT1/img', 50)
    imagenames = rm.read_imagename('../Data/N2DH-GOWT1/img')
    flattened = rm.image_flatten(imageread)
    img_df = rm.dataframe(flattened, imagenames)

    imageread_gt = pt.tiles('../Data/N2DH-GOWT1/gt/tif', 50)
    imagenames_gt = rm.read_imagename('../Data/N2DH-GOWT1/gt/tif')
    flattened_gt = rm.image_flatten(imageread_gt)
    gt_df = rm.dataframe(flattened_gt, imagenames_gt)

    main(img_df, gt_df, 0)
