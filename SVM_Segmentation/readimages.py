import os
import numpy
from skimage import io
import numpy as np
from numpy import asarray, ndarray
import pandas as pd


def read_image(path_of_imagefolder):
    """
    This function reads images from a given path.
    :param path_of_image: path of the image which has to be uploaded.
    :return:
    """
    image_list = []
    if not os.path.exists(path_of_imagefolder):
        raise FileNotFoundError
    for filename in os.listdir(path_of_imagefolder):
        img = io.imread(os.path.join(path_of_imagefolder, filename))
        image = {filename: np.asarray(img)}
        if img is not None:
            image_list.append(image)
    return image_list
#here also name and dataframe and also for the next function/function beneath

#***MAYBE: name the images, so that we can differentiate between them in the array.***
def image_flatten(image_list):
    """
    This function reads all images from a folder as arrays and arranges them into a list
    :param folder: Folder where the images are located.
    :return: A list of images as arrays
    """
    imagelist = image_list
    element = imagelist[1]
    for element[1] in imagelist:
        if element[1] is numpy.ndarray:
            flattened = ndarray.flatten(element)
            reshaped = flattened.reshape(1, -1)
    #image_list_dataframe = pd.DataFrame(image_list)
    return imagelist

##aktuelles Problem: durch assignen von Namen erhalten wir eine Liste von tuplen. Wir wollen über die Arrays,
# also Elemente von den Tuples iterieren und die arrays flatten, aber

imageread = read_image('../Data/N2DH-GOWT1/gt/jpg')
for element in imageread:
    imagelist = []
    array = element.values()
    array2 = asarray(array)
    flattened = ndarray.flatten(array2)
    print(flattened)


imageflatten = image_flatten(imageread)
#if imageflatten[1] is numpy.ndarray:
#print(imageflatten)




