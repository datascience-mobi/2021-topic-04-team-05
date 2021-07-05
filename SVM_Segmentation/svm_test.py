import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import cv2
import readimages as rm
import PCA




# functions need for the loss function
def distance_of_point_to_hyperplane(w, x, y):
    return 1 - y * (np.dot(x, w))

print(distance_of_point_to_hyperplane(1,3,5))