import numpy as np
import cv2

#reading in images
img1 = cv2.imread('Synthetic Image 1.png')
img1a = cv2.imread('Synthetic Image 1a.png')
img2 = cv2.imread('Synthetic Image 2.png')

#defining function with variables for segmented picture and ground truth
def dice_coefficient(imgt, imgp):  # t = ground truth, p = SVM prediction
    assert imgt.dtype == np.bool
    #image as array
    assert imgp.dtype == np.bool
    intersection = np.logical_and(imgt, imgp)
    union = imgt.sum() + imgp.sum()
    if intersection + union == 0:
        return("dice cannot be calculated - no intersection")
    else:
        dice = (2 * intersection) / (union + intersection)
        return dice

dice_coefficient(img1, img1)
