import readimages as rm
import numpy as np
import math


def oneD_array_to_twoD_array_df(ExtendedDataFrame):
    for i in range(len(ExtendedDataFrame)):
        twoDarray = np.stack(ExtendedDataFrame.iloc[i], axis=0)
        a = int(math.sqrt(len(twoDarray)))
        twoDarray = twoDarray.reshape(a, a)
    return twoDarray


def oneD_array_to_twoD_array(array):
    twoDarray = np.stack(array, axis=0)
    a = int(math.sqrt(len(twoDarray)))
    twoDarray = twoDarray.reshape(a, a)
    return twoDarray

