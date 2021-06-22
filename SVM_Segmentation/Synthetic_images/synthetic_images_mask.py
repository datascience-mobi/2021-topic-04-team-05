import os
import random
import numpy
import numpy as np
import errno
import csv
from PIL import Image

def compimage(path_of_mask, path_of_background):

#foreground
    if os.path.exists(path_of_mask):
        if os.path.splitext(path_of_mask)[1].lower() == '.png':
            mask = Image.open(path_of_mask)
            #getting the information wether a pixel is transparent or not
            mask_alpha = np.array(mask.getchannel(3))
        else:
            print('Only png files can be used as mask.')
    else:
        print('Wrong foreground image-path: {}'.format(path_of_mask))
    if np.all(mask) != 0:
        print('If mask has no transparent portion it is not an appropriate mask.: {}'.format(mask))

#background
    if os.path.exists(path_of_background):
        if os.path.splitext(path_of_background)[1].lower() in ['.png', '.jpg', 'jpeg']:
            background = Image.open(path_of_background)
            background_array = np.asarray(background)
            composedimage = [[0] * background_array.shape[0] for _ in range(background_array.shape[1])]
            for i in enumerate(mask_alpha):
                for j in enumerate(background_array):
                    if i != 0:
                        composedimage += background_array[j]
                    else:
                        composedimage += mask_alpha[i]
        else:
            print('Only png or jpg files can be used as background.')
    else:
        print('Wrong background image-path: {}'.format(path_of_background))

    return  composedimage

compimage('Pictures/Foreground/1.png', 'Pictures/Background/1.jpg')

#img = Image.open('Pictures/Background/1.jpg')
#print(img)
#print(img.shape)

#aktuelles Problem/Stand: png Bilder sind 3-dimensional --> wir müssen, dritte Dimension wegbekommen, da unwichtig?
#andere Fragen/Probleme/Tasks: -neuer Folder für generierte Bilder
                              #- Größe der Bilder bzw, wie können wir die maske an einen random Ort setzen