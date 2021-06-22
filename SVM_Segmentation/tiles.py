import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import image_slicer
import math
from matplotlib.image import imread

def tiles(image, number):
    list = [] #creates an empty list, which will be filled during the iterations with the tiles
    M = image.shape[0]//number #M=total number of lines, divided through the number = # of tiles
    N = image.shape[1]//number #N=total number of columns, divided through the number = # of tiles
    for x in range(0, image.shape[0], M): #iterations, starting in the first pixel until the last line,
        # and the amount of steps is the M which was already calculated
        for y in range(0, image.shape[1], N): #iterations, starting in the first pixel until the last column,
        # and the amount of steps is the N which was already calculated
            list.append([image[x:x + M, y:y + N]]) #the already created, empty list, is appended with the tile which
        # is "cut" during each iteration
    return list