import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import cv2
import readimages as rm
import preparation_PCA
import imagecodecs


# functions need for the loss function
def distance_of_point_to_hyperplane(w, x, y):
    """

    :param w:
    :param x:
    :param y:
    :return:
    """

    distance_hyperplane = 1 - y * (np.dot(x, w))
    return distance_hyperplane


def loss_function(w, x, y, C: float = 1e5):
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
    separation = [0 if i < 0 else i for i in separation]
    hinge_loss = C * (np.sum(separation) / N)

    # calculate loss
    loss = 1 / 2 * np.dot(w, w) + hinge_loss
    return loss


# functions needed for the gradient
def distance_of_point_to_sv(w, x, y, C: float = 1e5):
    """

    :param index:
    :param w:
    :param x:
    :param y:
    :param C:
    :return:
    """
    distance_sv = w - (C * y * x)
    return distance_sv


# calculating the gradient
def lagrange(w, x, y):
    """
    This function calculates the gradient of loss, which is then to be minimized.
    :param x: An array with the features of the samples.
    :param w: The vector of the feature weights.
    :param y: A dataframe with the labels of the samples.
    :return: A vector representing the gradient of the loss. #vector?
    """
    if x.shape == (1,):
        x = x[0]
    y = y[0]
    separation = distance_of_point_to_hyperplane(w, x, y)
    # separation is an 1Darray, if 1 feature and an multidimensional array if more features
    # in this array every element is the distance of a pixel of one feature/image
    if separation.shape == (1,):
        separation = separation[0]
    separation_df = np.asarray(separation)
    rows = separation_df.shape[0]
    gradient = 0
    for q in range(0, rows):
        # for correctly classified
        if q < 0:
            qi = w
        # for falsely classified points
        else:
            qi = distance_of_point_to_sv(w, x, y)
        df = pd.DataFrame(qi)
        for qi in range(0, df.shape[1]):
            gradient += qi
            # calculate average of distances
            gradient = gradient / len([y])
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
    weights = np.zeros(features.shape[1])
    power = 0
    unbounded_upper_value = float("inf")
    stoppage_criterion = 0.01  # in percent
    for epoch in range(1, maximum_epochs):
        # shuffle prevents the same x & y being take for several rounds
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
    :param gt_path: The path of the ground truth images.
    :return: The vector of the feature weights.
    """

    # read in microscopic images
    imageread = rm.read_image(img_path)
    image_PCA = preparation_PCA.convert_pca(imageread, 0.75)
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
    #main('../Data/test/img', '../Data/test/gt')

    imageread = rm.read_image('../Data/test/img')
    image_PCA = preparation_PCA.convert_pca(imageread, 0.75)
    # normalizing microscopic images
    normalizedimg = []
    for i in range(0, len(imageread)):
        pixelsimg = imageread[i].astype('float32')
        if pixelsimg.max() > 0:
            normalimg = pixelsimg / pixelsimg.max()
            normalizedimg.append(normalimg)
        else:
            normalizedimg.append(pixelsimg)
    imagenames = rm.read_imagename('../Data/test/img')
    imageflattended = rm.image_flatten(image_PCA)

    X = rm.dataframe(imageflattended, imagenames)
    X.insert(loc=len(X.columns), column='intercept', value=1)
    X = X.iloc[:, 0:5]

    #print(X)

    # read in ground truth images
    gtread = rm.read_image('../Data/test/gt')

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
    gtnames = rm.read_imagename('../Data/test/gt')
    thresholded_and_normalized_flattened = rm.image_flatten(normalizedgt)

    y = rm.dataframe(thresholded_and_normalized_flattened, gtnames)
    y = y.iloc[:, 0:5]
    y_labels = y.replace(0, -1)

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
        y_train = y_labels.iloc[result[0]]
        y_test = y_labels.iloc[result[1]]

    X_train = X_train.transpose()
    y_train = y_train.transpose()
    X_test = X_test.transpose()
    y_test = y_test.transpose()



        #print(X_test)

        #print(y_train)

        #print(X_train)

        #print(y_test)

        #print(distance_of_point_to_hyperplane(7, X_train, y_train))

        #print(loss_function(X_train, 7, y_train))

        #stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy())



        #separation = distance_of_point_to_hyperplane(7, X_train, y_train)
        #separation_df = pd.DataFrame(separation)
        #gradient = 0
        #columns = separation_df.shape[1]
        #print(separation_df)

        #C = 1e5
        #list = []
        #for index in range(0, columns):
            #distance_sv = w - (C * y_train.iloc[0, index] * X_train.iloc[0, index])
            #print(distance_sv)
            #list.append(distance_sv)
        #rowname = X_train.index[0]
        #rowname_list = (rowname)
        #print(rowname_list)
        #print(rowname)
        #df_final = pd.DataFrame(list)
        #df_transposed = df_final.transpose()
        #df_transposed2 = df_transposed.rename(index={0: (f'{rowname}')})
        #print(df_transposed2)

    #separation = distance_of_point_to_hyperplane(w, x, y)
    #separation_df = pd.DataFrame(separation)
    #columns = separation_df.shape[1]
    #distances_sv_list = []
    #gradient = 0
    #for q in range(0, columns):
        # for correctly classified
        #if q < 0:
            #qi = w
        # for falsely classified points
        #else:
            #index = q
            #qi = distance_of_point_to_sv(index, w, x, y)
        #distances_sv_list.append(qi)
        #rowname = x.index[0]
        #df = pd.DataFrame(distances_sv_list)
        #df_transposed = df.transpose()
        #df_renamed = df_transposed.rename(index={0: (f'{rowname}')})
        #qi_list = []
        #for qi in range(0, df_renamed.shape[1]):
            #gradient += qi
            # calculate average of distances
            #gradient = gradient / len(y)
            #qi_list.append(gradient)
            #df_qi = pd.DataFrame(qi_list)
            #df_qi_transposed = df_qi.transpose()
            #df_qi_renamed = df_qi_transposed.rename(index={0: (f'{rowname}')})
    #print(df_qi_renamed)

    features = X_train
    labels = y_train

    learning_rate: float = 1e-6
    #number of features starting with 0, so for one feature it is 0, for 2 it is 1 etc.
    number_of_features = 0

    maximum_epochs = 5000
    array_of_weights = np.zeros(features.shape[1])
    power = 0
    unbounded_upper_value = float("inf")
    stoppage_criterion = 0.01  # in percent
    for epoch in range(1, maximum_epochs):
        #shuffle prevents the same x & y being taken for several rounds
        x, y = shuffle(features, labels)
        i = 0
        if number_of_features != 0:
            for i in [0, x.shape[1]]:
                end = i + number_of_features
                x = x.values[:, [i, end]]
                i += number_of_features
        else:
            for k in range(0, x.shape[1]):
                x = x.values[:, [k]]
        for j in range(0, y.shape[1]):
            y = y.values[:, [j]]
            list = []
            df_of_weights = ()
            for ind, value in enumerate(x):
                upward_slope = lagrange(array_of_weights, value, y[ind])
                array_of_weights = array_of_weights - (learning_rate * upward_slope)
            if epoch == pow(2, power) or epoch == maximum_epochs - 1:
                if array_of_weights.shape == (1,):
                    array_of_weights = array_of_weights[0]
                loss = loss_function(array_of_weights, features.values, labels.values)
                print("{}. epoch: current loss is {}.".format(epoch, loss))
                # stoppage criterion to stop at convergence
                deviance = abs(unbounded_upper_value - loss)
                # if cost no longer changes, stop gradient decend
                if stoppage_criterion * unbounded_upper_value > deviance:
                    print(array_of_weights)
                unbounded_upper_value = loss
                power += 1
    print(array_of_weights)



        # train the model
        #W = stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy())
        #print("The weights vector is: {}".format(W))
