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
    """
    This function cuts each input image into n tiles of size NxN and then asserts the mean intensity value of all original
    pixels to that tile.
    :param image_path: path of the images
    :param number: amount of tiles that the image should be cut into
    :return: a list of arrays representing the tiles of the input images; the intensity value of each tile equals the
    mean intensity value of each tile
    """

    images = rm.read_image(image_path)
    names = rm.read_imagename(image_path)
    list_of_arrays = []
    for index, image in enumerate(images):
        list = []
        M = image.shape[0] // number
        N = image.shape[1] // number
        for x in range(0, image.shape[0], M):
            for y in range(0, image.shape[1], N):
                list.append([image[x:x + M, y:y + N]])
        list_mean = []
        for i in range(0, len(list)):
            mean = np.mean(list[i])
            list_mean.append(mean)
        array_mean = np.asarray(list_mean)
        # convert 1D array to 2D array
        two_d_array_mean = pc.one_d_array_to_two_d_array(array_mean)
        list_of_arrays.append(two_d_array_mean)
    return list_of_arrays


tiles('../Data/N2DH-GOWT1/gt/tif', 50)
