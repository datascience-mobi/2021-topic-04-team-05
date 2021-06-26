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
    This function reads all images from a folder as arrays and arranges them into a list
    :param folder: Folder where the images are located.
    :return: A list of images as arrays
    """
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
    pd.DataFrame(image_list)





#Tests

imageread = read_image('../Data/N2DH-GOWT1/gt/jpg')
imagenames = read_imagename('../Data/N2DH-GOWT1/gt/jpg')
#imageflattened = image_flatten(imageread)

print(imageread[3])
print(imagenames[3])

#print(imageflattened)




