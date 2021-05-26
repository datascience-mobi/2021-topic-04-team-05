import numpy as np
import cv2

#reading in images
img1 = cv2.imread('Synthetic Image 1.png')
img1a = cv2.imread('Synthetic Image 1a.png')
img2 = cv2.imread('Synthetic Image 2.png')

#designing reusable function
def dice_coefficient(imgt, imgp):  # t = ground truth, p = SVM prediction
    assert imgt.dtype == np.bool #the images with type array are converted to type bool
    assert imgp.dtype == np.bool #the images with type array are converted to type bool
    intersection = np.logical_and(imgt, imgp) #compute the truth value of x1 AND x2 element-wise = sums all pixels where both gt and pred have the value 'true'
    union = imgt.sum() + imgp.sum() #compute the truth value of x1 OR x2 element-wise = sums all pixels where either gt or pred (or both) have the value 'true'
    if intersection + union == 0:
        return ('dice cannot be calculated - no intersection') #because it is mathematically not allowed to divide by 0, which would happen if gt and pred don't intersect
    else:
        dice = (2 * intersection) / (union + intersection) #using the dice formula to calculate the dice IF gt and pred intersect
        return dice #print out dice

#using function dice_coefficient to compute the dice score
dice_coefficient(img1, img1)
