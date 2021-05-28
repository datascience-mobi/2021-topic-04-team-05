import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.image import imread
from scipy import misc, ndimage

img = imread('SyntheticImage1.png')
print(type(img))

imguint8 = img.astype(np.uint8)
intersection = np.logical_and(imguint8, imguint8)
dice = (2 * intersection.sum())/(imguint8.sum() + imguint8.sum())
print(dice)

img1 = imread('SyntheticImage2.png')

img1uint8 = img1.astype(np.uint8)
intersection = np.logical_and(img1uint8, imguint8)
dice = (2 * intersection.sum())/(img1uint8.sum() + imguint8.sum())
print(dice)



#______________________________________________________________________________________________-

#Load and show an image with Pillow
from PIL import Image

#Load the image
img = Image.open('SyntheticImage1.png')

#Get basic details about the image
print(img.format)
print(img.mode)
print(img.size)

#show the image
img.show()

intersection = np.logical_and(img1, img1)
union = img1.sum() + img1.sum() #np.sum(img1() + img1())
dice = (2 * intersection) / (union + intersection)
print (dice)


#reading in images
img1 = cv2.imread('Synthetic Image 1.png') #as array
img1a = cv2.imread('Synthetic Image 1a.png')
img2 = cv2.imread('Synthetic Image 2.png')

def dice_coefficient(imgt, imgp):
    imgt = np.dtype(bool) #oder imgt.astype('bool')
    imgp = np.dtype(bool)
    intersection = np.logical_and(imgt, imgp) #compute the truth value of x1 AND x2 element-wise = sums all pixels where both gt and pred have the value 'true'
    union = np.logical_or(imgt, imgp) #compute the truth value of x1 OR x2 element-wise = sums all pixels where either gt or pred (or both) have the value 'true'
    dice = (2 * intersection.sum) / (union + intersection) #using the dice formula to calculate the dice IF gt and pred intersect
    return dice #print out dice

dice_coefficient(img1, img1)



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


def dice_coefficient(imgt, imgp):
    imgt = np.dtype(np.bool)
    imgp = np.dtype(np.bool)
    intersection = np.logical_and(imgt, imgp) #compute the truth value of x1 AND x2 element-wise = sums all pixels where both gt and pred have the value 'true'
    union = imgt.sum() + imgp.sum() #compute the truth value of x1 OR x2 element-wise = sums all pixels where either gt or pred (or both) have the value 'true'
    if intersection + union == 0:
        return ('dice cannot be calculated - no intersection') #because it is mathematically not allowed to divide by 0, which would happen if gt and pred don't intersect
    else:
        dice = (2 * intersection) / (union + intersection) #using the dice formula to calculate the dice IF gt and pred intersect
        return dice #print out dice

dice_coefficient(img1, img1)
