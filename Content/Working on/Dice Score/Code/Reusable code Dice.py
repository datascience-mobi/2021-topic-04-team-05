import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.image import imread
from scipy import misc, ndimage

def dice_score (pred,gt):
    preduint8 = pred.astype(np.uint8)
    gtuint8 = gt.astype(np.uint8)
    intersection = np.logical_and(preduint8, gtuint8)
    dice = (2 * intersection.sum()) / (preduint8.sum() + gtuint8.sum())
    print(dice)


img1 = imread ('SyntheticImage1.png')
img2 = imread ('SyntheticImage2.png')

dice_score (img1, img2)