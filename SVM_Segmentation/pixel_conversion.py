import numpy as np
import math


def one_d_array_to_two_d_array_df(extended_data_frame):
    """
       A function to convert a 1D Array back to 2D Array from data frame.
       :param: Data Frame of flattened arrays
       :return: Data frame of 2D Array
       """
    for i in range(len(extended_data_frame)):  # the length of dataframe
        two_d_array = np.stack(extended_data_frame.iloc[i], axis=0)
        n = int(math.sqrt(len(two_d_array)))  # converting the array into NxN 2D Array
        two_d_array = two_d_array.reshape(n, n)
    return two_d_array


def one_d_array_to_two_d_array(array):
    """
           A function to convert a 1D Array back to 2D Array (one array)
           :param: Flattened arrays
           :return: One 2D Array
           """
    two_d_array = np.stack(array, axis=0)
    a = int(math.sqrt(len(two_d_array)))
    two_d_array = two_d_array.reshape(a, a)
    return two_d_array

