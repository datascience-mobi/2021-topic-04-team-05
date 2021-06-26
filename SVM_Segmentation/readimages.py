import os
import numpy
from skimage import io
import numpy as np
from numpy import asarray, ndarray
import pandas as pd


def read_image(path_of_imagefolder):
    """
    This function reads images from a given path and puts them into a list of arrays.
    :param path_of_imagefolder: path of the images
    :return: list of np.ndarrays
    """
    image_list = []
    if not os.path.exists(path_of_imagefolder):
        raise FileNotFoundError
    for filename in os.listdir(path_of_imagefolder):
        img = io.imread(os.path.join(path_of_imagefolder, filename))
        image = np.asarray(img)
        if img is not None:
            image_list.append(image)
    return image_list

def read_imagename(path_of_imagefolder):
    """
    This function creates a list of the filenames in a directory.
    :param path_of_image: path of the folder with the files the filenames are wanted from
    :return: a list of the filenames
    """
    name_list = []
    if not os.path.exists(path_of_imagefolder):
        raise FileNotFoundError
    for filename in os.listdir(path_of_imagefolder):
        if filename is not None:
            name_list.append(filename)
    return name_list

def image_flatten(image_list):
    """
    This function flattens all arrays of a list.
    :param image_list: A list of images as arrays.
    :return: A list of flattened arrays.
    """
    if type(image_list) != 'list':
        raise TypeError("Input has to be of type 'list'.")
    imagelist_flattened = []
    for element in image_list:
        if element is not None:
            flattened = ndarray.flatten(element)
            reshaped = flattened.reshape(1, -1)
            imagelist_flattened.append(reshaped)
    return imagelist_flattened

def dataframe(image_list, name_list):
    """
    This function creates a dataframe from a list of flattened arrays (as rows) and their names as rownames.
    :param image_list: A list of flattened arrays.
    :param name_list: A list of names, in the same order as the arrays in the image_list.
    :return:
    """
    if len(image_list) != len(name_list):
        raise ValueError("Lists have to be of the same length.")
    dataframe_images = pd.DataFrame()
    i = 0
    for i in range(0, len(image_list)):
        array = image_list[i]
        if array.shape[0] != 1:
            raise ValueError("Array has to be of the shape (1, x).")
        element = pd.DataFrame(array)
        i += 1
        dataframe_images = dataframe_images.append(element)
    dataframe_images = dataframe_images.set_axis(name_list, axis=0)

    return dataframe_images

def fuse_dataframes(dataframe1, dataframe2, dataframe3):
    """
    Fusing 3 dataframes, with similiar content, by inserting the first row of dataframe2 and after that of dataframe 3,
    after the first row of dataframe 1. The name of the dataframe is added to the according rownames to distinguish
    between the different samples.

    :param dataframe1: dataframe 1, with pixels as features and rows as samples.
    :param dataframe2: dataframe 2, with pixels as features and rows as samples.
    :param dataframe3: dataframe 3, with pixels as features and rows as samples.
    :return: dataframe composed of all 3 input-dataframes
    """
    if dataframe1.shape != dataframe2.shape != dataframe3.shape:
        raise ValueError("Dataframes have to be the same shape.")




#Tests

imageread = read_image('../Data/N2DH-GOWT1/gt/jpg')
imagenames = read_imagename('../Data/N2DH-GOWT1/gt/jpg')
imageflattened = image_flatten(imageread)
dataframe(imageflattened, imagenames)




