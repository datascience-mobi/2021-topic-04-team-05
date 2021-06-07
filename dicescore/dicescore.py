#DICE SCORE
#python library for working with arrays
import numpy as np
#support for opening, manipulating, and saving images
import PIL

#reading the images and converting to arrays
from matplotlib.image import imread

img1 = np.asarray(imread('man_seg21_totest.png'))
#img1 = np.asarray(PIL.Image.open('man_seg21_totest.png'))

def dice_score (pred, gt):
    dice = np.sum(pred[gt == pred]) * 2.0 / (np.sum(gt) + np.sum(pred))
    print (dice)

dice_score (img1, img1)