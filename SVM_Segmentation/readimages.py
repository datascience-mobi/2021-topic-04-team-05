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
    #if type(image_list) != 'list':
        #raise TypeError("Input has to be of type 'list'.")
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

def fuse_dataframes(dataframe1, name1, dataframe2, name2, dataframe3, name3):
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
    #if type(dataframe1) and type(dataframe2) and type(dataframe3) != type(pd.DataFrame):
        #raise TypeError("Input has to be of type 'pandas.core.frame.DataFrame'.")
    fused_dataframe = pd.DataFrame()
    row = 0
    dataframe1 = dataframe1.rename(index=lambda s: s + str(name1))
    dataframe2 = dataframe2.rename(index=lambda s: s + str(name2))
    dataframe3 = dataframe3.rename(index=lambda s: s + str(name3))
    for row in range(0,len(dataframe1)):
        for dataframe in [dataframe1, dataframe2, dataframe3]:
            fused_dataframe = fused_dataframe.append(dataframe.iloc[row, :])
        row += 1
    return fused_dataframe

#Test
dataframe1 = pd.DataFrame([['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H'], ['I', 'J', 'K', 'L'], ['M', 'N', 'O', 'P']])
dataframe2 = pd.DataFrame([['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', '11', '12'], ['13', '14', '15', '16']])
dataframe3 = pd.DataFrame([['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h'], ['i', 'j', 'k', 'l'], ['m', 'n', 'o', 'p']])

dataframe1_names = ['A', 'B', 'C', 'D']
dataframe2_names = ['1', '2', '3', '4']
dataframe3_names = ['a', 'b', 'c', 'd']

d1 = dataframe1.set_axis(dataframe1_names, axis=0)
d2 = dataframe2.set_axis(dataframe2_names, axis=0)
d3 = dataframe3.set_axis(dataframe3_names, axis=0)


#print(fuse_dataframes(d1, 'd1', d2, 'd2', d3, 'd3'))

#Tests

imageread1 = read_image('../Data/N2DH-GOWT1/img')
#print(imageread1)
imagenames1 = read_imagename('../Data/N2DH-GOWT1/img')
#print(imagenames1)
imageflattened1 = image_flatten(imageread1)
#print(imageflattened1)
data1 = dataframe(imageflattened1, imagenames1)
#print(data1)

imageread2 = read_image('../Data/N2DL-HeLa/img')
#print(imageread2)
imagenames2 = read_imagename('../Data/N2DL-HeLa/img')
#print(imagenames2)
imageflattened2 = image_flatten(imageread2)
#print(imageflattened2)
data2 = dataframe(imageflattened2, imagenames2)
#print(data2)

imageread3 = read_image('../Data/NIH3T3/img')
#print(imageread3)
imagenames3 = read_imagename('../Data/NIH3T3/img')
#print(imagenames3)
imageflattened3 = image_flatten(imageread3)
#print(imageflattened3)
data3 = dataframe(imageflattened3, imagenames3)
#print(data3)

fuse_dataframes(data1, d1, data2, d2, data3, d3)



