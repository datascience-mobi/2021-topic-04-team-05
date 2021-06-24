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