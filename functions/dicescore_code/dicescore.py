# DICE SCORE
# python library for working with arrays
import numpy as np
# support for opening, manipulating, and saving images

# reading the images and converting to arrays
from matplotlib.image import imread


def dice_score(pred, gt):
    dice = np.sum(pred[gt == pred]) * 2.0 / (np.sum(gt) + np.sum(pred))
    print(dice)


if __name__ == '__main__':
    img1 = np.asarray(imread('man_seg21testing.png'))
    # img1 = np.asarray(PIL.Image.open('man_seg21_totest.png'))
    dice_score(img1, img1)