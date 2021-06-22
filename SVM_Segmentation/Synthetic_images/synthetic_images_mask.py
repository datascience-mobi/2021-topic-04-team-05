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