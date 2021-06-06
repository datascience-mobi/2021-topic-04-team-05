import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.image import imread
from scipy import misc, ndimage

def dice_score (pred,gt):
    preduint8 = pred.astype(np.uint8) #convert prediction image to different array type: uint8 = 8 bit unsigned integer (range of values: 0 to 255)
    gtuint8 = gt.astype(np.uint8) #convert ground truth to array type uint8
    intersection = np.logical_and(preduint8, gtuint8) #calculate the intersection of both images, the  truth value of x1 AND x2 element-wise
    dice = (2 * intersection.sum()) / (preduint8.sum() + gtuint8.sum()) #calculate dice (2*intersection/union)
    print(dice)

img1 = imread ('SyntheticImage1.png') #import image
img2 = imread ('SyntheticImage2.png') #import image

dice_score (img1, img2)