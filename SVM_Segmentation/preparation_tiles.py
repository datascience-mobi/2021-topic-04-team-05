import os
import cv2
import numpy as np
import pandas as pd
import math
from skimage import io
from matplotlib import pyplot as plt
from SVM_Segmentation import readimages as rm
from SVM_Segmentation import pixel_conversion as pc


def tiles(image_path, number):
    images = rm.read_image(image_path)
    names = rm.read_imagename(image_path)
    list_of_arrays = []
    for index, image in enumerate(images):
        list = [] #creates an empty list, which will be filled during the iterations with the tiles
        M = image.shape[0]//number #M=total number of lines, divided through the number = # of tiles
        N = image.shape[1]//number #N=total number of columns, divided through the number = # of tiles
        for x in range(0, image.shape[0], M): #iterations, starting in the first pixel until the last line,
            # and the amount of steps is the M which was already calculated
            for y in range(0, image.shape[1], N): #iterations, starting in the first pixel until the last column,
            # and the amount of steps is the N which was already calculated
                list.append([image[x:x + M, y:y + N]]) #the already created, empty list, is appended with the tile which
        # is "cut" during each iteration
        list_mean = []
        for i in range(0, len(list)):
            mean = np.mean(list[i])
            list_mean.append(mean)
        array_mean = np.asarray(list_mean)
        twod_array_mean = pc.oneD_array_to_twoD_array(array_mean)
        list_of_arrays.append(twod_array_mean)
        for index, name in enumerate(names):
            filename = name
            path = (f'../Data/tiles/{filename}.tif'
            cv2.imwrite(path, twod_array_mean)
    return list_of_arrays


tiles('../Data/N2DH-GOWT1/gt/tif', 50)