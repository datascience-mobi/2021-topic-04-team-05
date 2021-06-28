# DICE SCORE
# python library for working with arrays
import numpy as np
# support for opening, manipulating, and saving images

# reading the images and converting to arrays
from matplotlib.image import imread


def dice_score(pred, gt):
    """
    This function calculates the similiarity between two arrays.
    :param pred: an array of predicted labels
    :param gt: the ground truth of this array
    :return: a value between 0 and 1, describing the similiarity between those arrays. 1 is the dice score of similiar arrays.
    """
    dice = np.sum(pred[gt == pred]) * 2.0 / (np.sum(gt) + np.sum(pred))
    print(dice)

#führt Code nur aus, wenn dicescore.py direkt gerunnt wird & nicht wenn es in anderes pythonfile importiert wird
if __name__ == '__main__':
    img1 = np.asarray(imread('../SVM_Segmentation/Synthetic_images/GeneratedImages/mask4.png'))
    img2 = np.asarray(imread('../SVM_Segmentation/Synthetic_images/GeneratedImages/mask5.png'))
    #img1 = np.asarray(PIL.Image.open('man_seg21_totest.png'))
    dice_score(img1, img2)