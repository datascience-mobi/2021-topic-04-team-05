import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import image_slicer
import math
from matplotlib.image import imread

im=cv2.imread('/Users/laurasanchis/PycharmProjects/svm/Data/N2DH-GOWT1/gt/man_seg21.tif')

def tiles(image, number):
    list = []
    M = image.shape[0]//number #number ist menge der tiles
    N = image.shape[1]//number
    for x in range(0, image.shape[0], M):
        for y in range(0, image.shape[1], N):
            list.append([image[x:x + M, y:y + N]])
    return list

tiles1 = tiles(im, 2)
print(tiles1)