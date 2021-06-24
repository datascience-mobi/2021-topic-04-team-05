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



