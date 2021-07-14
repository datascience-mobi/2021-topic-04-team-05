# DICE SCORE
# python library for working with arrays
import numpy as np
# support for opening, manipulating, and saving images

# reading the images and converting to arrays
from matplotlib.image import imread


def dice_score(pred, gt):
    """
    This function calculates the similarity between two arrays.
    :param pred: an array of predicted labels
    :param gt: the ground truth of this array
    :return: a value between 0 and 1, describing the similarity between those arrays. 1 is the dice score of similar arrays.
    """
    dice = np.sum(pred[gt == pred]) * 2.0 / (np.sum(gt) + np.sum(pred))
    print(dice)


if __name__ == '__main__':
    img1 = np.asarray(imread('../SVM_Segmentation/Synthetic_images/synthetic_masks/mask4.png'))
    img2 = np.asarray(imread('../SVM_Segmentation/Synthetic_images/synthetic_masks/mask5.png'))
    dice_score(img1, img2)
