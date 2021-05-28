#DICE SCORE
#python library for working with arrays
import numpy as np
#support for opening, manipulating, and saving images
import PIL
#reading the images and converting to arrays
bild1 = np.asarray(PIL.Image.open('Synthetic Image 1 .png'))
bild1a = np.asarray(PIL.Image.open('Synthetic Image 1a.png'))
bild2 = np.asarray(PIL.Image.open('Synthetic Image 2.png'))

#implementing the dice score
#comparing 2 same images
dice_same = np.sum(bild1[bild1a==bild1])*2.0 / (np.sum(bild1a) + np.sum(bild1))
#comparing two different images
dice_different = np.sum(bild1[bild2==bild1])*2.0 / (np.sum(bild1) + np.sum(bild2))
#printing the dice score
print(dice_same)
print(dice_different)