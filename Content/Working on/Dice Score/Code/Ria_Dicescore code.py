#imports packages
import numpy as np
import cv2 as cv2

#designing the dice coefficient pt.1
#loading the images (true = ground truth image; pred = predicted segmentation by SVM)
img1_true = cv2.imread('path to image') #reads image into numpy array (2D matrix)
img1_pred = cv2.imread('path to image')

#convert image into type bool
img1t = img1_true.astype(bool)
img1p = img1_pred.astype(bool)

#designing the dice coefficient pt.2
img2_true = imread('path to image') #import image
img2_pred = imread('path to image') #import image

#convert image into type bool
img2t = np.asarray(img1_true).astype(np.bool) #converts image to array and then to bool
img2p = np.asarray(img1_pred).astype(np.bool) #converts image to array and then to bool


#defining the variables 'intersection' and 'union'
intersection = np.logical_and(img1t, img1p) #Compute the truth value of x1 AND x2 element-wise
union = img1t.sum() + img1p.sum()

#calculating the dice
if intersection + union == 0:
    print('dice cannot be calculated - no intersection')
else:
    dice = (2*intersection)/(union + intersection)
    print(dice)

#reusable code
# import images (prediction & ground truth) as arrays

# compute dice score
def dice_coefficient(imgt, imgp):  # t = ground truth, p = SVM prediction
    assert imgt.dtype == np.bool
    assert imgp.dtype == np.bool
    intersection = np.logical_and(imgt, imgp)
    union = imgt.sum() + imgp.sum()
    if intersection + union == 0:
        return ('dice cannot be calculated - no intersection')
    else:
        dice = (2 * intersection) / (union + intersection)
        return dice

