import os
import random
import numpy
import numpy as np
import errno
import csv
from PIL import Image
from SVM_Segmentation import readimages

def compimage(path_of_mask, path_of_background):

def foreground(path_of_folder_with_mask):
    """

    :param path_of_folder_with_mask:
    :return:
    """
    mask_image_dataframe = readimages.read_image(path_of_imagefolder=path_of_folder_with_mask)

    for row in mask_image_dataframe:
        if not os.path.splitext(row)[1].lower() == '.png':
            raise (f'Only png files can be used as mask.')
        # getting the information whether a pixel is transparent or not
        mask_alpha = np.array(image.getchannel(3))
        if not np.any(mask_alpha) == 0:
            raise (f'If mask has no transparent portion it is not an appropriate mask.: {mask_alpha}')
            return foregroundimg

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
        return  backgroundimg


def compimage(foregroundimg, backgroundimg):
    # Rotate the foreground
    angle = random.randint(0, 359)
    foregroundrotated = sktrans.rotate(foregroundimg, angle)

    # Scale the foreground
    scale = random.uniform(0.1,1)
    new_size = (int(foregroundimg.shape[0] * scale), int(foregroundimg.shape[1] * scale))
    foregroundscaled = sktrans.resize(foregroundrotated, new_size)

    # Choose a random x,y position for the foreground
    max_pos = (backgroundimg.shape[0] - foregroundimg.shape[0], backgroundimg.shape[1] - foregroundimg.shape[1])
    if max_pos[0] <= 0 or max_pos[1] <= 0:
        raise (f'foreground is too big for the background.')
    position = (random.randint(0, max_pos[0]), random.randint(0, max_pos[1]))

    # Create a new foreground image as large as the background and paste it on top
    foregroundmask = Image.new('RGBA', backgroundimg.shape, color=(0, 0, 0, 0))
    foregroundnew =  foregroundmask.paste(foregroundscaled, position)

    composedimage = np.zeros(backgroundimg.shape)
    for i in foregroundnew.getchannel(3):
        for j in backgroundimg:
            for h in composedimage:
                if i > 0:
                    composedimage[h] += foregroundnew.getchannel(3)[i]
                else:
                    composedimage[h] += backgroundimg[j]
    return composedimage

#img = Image.open('Pictures/Background/1.jpg')
#print(img)
#print(img.shape)

#aktuelles Problem/Stand: png Bilder sind 3-dimensional --> wir müssen, dritte Dimension wegbekommen, da unwichtig?
#andere Fragen/Probleme/Tasks: -neuer Folder für generierte Bilder
                              #- Größe der Bilder bzw, wie können wir die maske an einen random Ort setzen