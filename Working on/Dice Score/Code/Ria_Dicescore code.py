#imports packages
import numpy as np
import cv2 as cv2

#designing the dice coefficient
#general idea: image --> type 'array' --> type 'bool'

# one way of doing it: 1.
#loading the images (true = ground truth image; pred = predicted segmentation by SVM)
img1_true = cv2.imread('path to image') #reads image into numpy array (2D matrix)
img1_pred = cv2.imread('path to image') #reads image into numpy array (2D matrix)

#convert image into type bool
img1t = img1_true.astype(bool)
img1p = img1_pred.astype(bool)

# other way of doing it: 2.
img2_true = imread('path to image') #import image as img2_true
img2_pred = imread('path to image') #import image as img2_pred

#convert image into type bool
img2t = np.asarray(img1_true).astype(np.bool) #converts image to array and then to bool
img2p = np.asarray(img1_pred).astype(np.bool) #converts image to array and then to bool


#defining the variables 'intersection' and 'union'
intersection = np.logical_and(img1t, img1p) #compute the truth value of x1 AND x2 element-wise = sums all pixels where both gt and pred have the value 'true'
union = img1t.sum() + img1p.sum() #compute the truth value of x1 OR x2 element-wise = sums all pixels where either gt or pred (or both) have the value 'true'

#calculating the dice
if intersection + union == 0: #because it is mathematically not allowed to divide by 0, which would happen if gt and pred don't intersect
    print('dice cannot be calculated - no intersection')
else:
    dice = (2*intersection)/(union + intersection) #using the dice formula to calculate the dice IF gt and pred intersect
    print(dice)

#reusable code
# import images (prediction & ground truth) as arrays

# compute dice score
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



