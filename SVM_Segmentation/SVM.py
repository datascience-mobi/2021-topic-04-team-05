import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
import random
import cv2

C =
learning_rate =

def loss_function (x,w,y):
    #calculate hinge loss
    N = x.shape[0]  #number of rows in x = number of samples
    separation = distance_of_point_to_hyperplane(w, x, y)  #calculates distance of x to hyperplane
    separation = [0 if i < 0 else i for i in separation] #all negative seperation values are replaced by 0
    hinge_loss = C * (np.sum(separation) / N)  # average loss because the whole Y is taken & it encompasses several samples

    # calculate loss
    loss = 1 / 2 * np.dot(w, w) + hinge_loss  # np.dot (W,W) ist gleich Betrag von w^2
    return loss

#functions we need for the gradient/for lagrange
def distance_of_point_to_hyperplane(w, x, y):
    return 1 - y * (np.dot(x, w))

def distance_of_point_to_sv(index, w, x, y):
    return w - (C * y[index] * x[index])

#lagrange
def lagrange (x: np.array,w,y): #ggf. x_sample, y_sample
    separation = distance_of_point_to_hyperplane(w, x, y)  #calculates distance of x to hyperplane
    gradient = 0
    for index, q in enumerate(separation):  # enumerate adds counter to the iterable to keep track of the number of items in the iterator; für jedes element in distance; ind = index & d ist zugehöriger distance wert
        if q < 0: #if d = negativ --> right classification
            qi = w
        else:
            qi = distance_of_point_to_sv(index, w, x, y) #für falsch klassifizierte: distanz zwischen punkt (xi,yi) und support vector (distanz von hyperplane zu SV ist W)
        gradient += qi
    gradient = gradient/len(y) #average of distances as len(y) is number of all trials
    return gradient

#minimize gradient using Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(features, labels):
    maximum_epochs = 5000 #an epoch indicates the number of passes of the entire training dataset the machine learning algorithm has completed
    weights = np.zeros(features.shape[1])  #creating array filled with zeros of the number of columns of our features (d.h. so viele wie features) dataset
    power = 0 #hoch
    unbounded_upper_value = float("inf") #acts as unbounded upper value for comparison for finding lowest values of something
    stoppage_criterion = 0.01  #in percent
    # stochastic gradient descent
    for epoch in range(1, maximum_epochs):
        x, y = random.shuffle(features, labels) #shuffle to prevent repeating update cycles; Stichproben von features & outputs werden rausgezogen (jede Runde neu)
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


def init():
    # read dataset
    # X = data.features
    # y = data.labels

    # read in normal images
    # rm = readimages.py
    imageread = rm.read_image('../Data/N2DH-GOWT1/img')  # Bilder eines Ordners in Liste mit 2D arrays
    normalizedimg = []
    for i in range(0, len(imageread)):
        pixelsimg = imageread[i].astype('float32')
        if pixelsimg.max() > 0:
            normalimg = pixelsimg / pixelsimg.max()
            normalizedimg.append(normalimg)
        else:
            normalizedimg.append(pixelsimg)
    imagenames = rm.read_imagename('../Data/N2DH-GOWT1/img')  # Liste mit Namen der Bilder
    imageflattended = rm.image_flatten(imageread)
    X = rm.dataframe(imageflattended, imagenames)

    # read in gt images
    gtread = rm.read_image('../Data/N2DH-GOWT1/gt/jpg')  # Bilder eines Ordners in Liste mit 2D arrays

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
    gtnames = rm.read_imagename('../Data/N2DH-GOWT1/gt/jpg')  # Liste mit Namen der Bilder
    thresholded_and_normalized_flattened = rm.image_flatten(normalized)
    y = rm.dataframe(thresholded_and_normalized_flattened, gtnames)  # ground truths

