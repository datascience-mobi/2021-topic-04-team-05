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
import cv2
import array_to_img as ai


def tiles(image, number):
    array = [] #creates an empty list, which will be filled during the iterations with the tiles
    M = image.shape[0]//number #M=total number of lines, divided through the number = # of tiles
    N = image.shape[1]//number #N=total number of columns, divided through the number = # of tiles
    for x in range(0, image.shape[0], M): #iterations, starting in the first pixel until the last line,
        # and the amount of steps is the M which was already calculated
        for y in range(0, image.shape[1], N): #iterations, starting in the first pixel until the last column,
        # and the amount of steps is the N which was already calculated
            array.append([image[x:x + M, y:y + N]]) #the already created, empty list, is appended with the tile which
        # is "cut" during each iteration
    list_mean = []
    for i in range(0, len(array)):
        mean = np.mean(array[i])
        list_mean.append(mean)
    array_mean = np.asarray(list_mean)
    return array_mean

if __name__ == '__main__':
    def oneD_array_to_twoD_array(oneDarray):
        twoDarray = np.stack(oneDarray, axis=0)
        a = int(math.sqrt(len(twoDarray)))
        twoDarray = twoDarray.reshape(a, a)
        return twoDarray

    gtread = io.imread('../Data/test/gt/man_seg01.jpg')
    gttiles = tiles(gtread, 50)
    gttwod = oneD_array_to_twoD_array(gttiles)
    plt.imshow(gttwod)
    plt.show()