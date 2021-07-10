import readimages as rm
import numpy as np
import math


def oneD_array_to_twoD_array_df(ExtendedDataFrame):
    """
       A function to convert a 1D Array back to 2D Array.
       :param Data Frame of Arrays:
       :return:
       """
    for i in range(len(ExtendedDataFrame)): #the length of dataframe
        twoDarray = np.stack(ExtendedDataFrame.iloc[i], axis=0)
        a = int(math.sqrt(len(twoDarray))) #converting the array into NxN 2D Array
        twoDarray = twoDarray.reshape(a, a)
    return twoDarray


def oneD_array_to_twoD_array(array):
    twoDarray = np.stack(array, axis=0)
    a = int(math.sqrt(len(twoDarray)))
    twoDarray = twoDarray.reshape(a, a)
    return twoDarray

