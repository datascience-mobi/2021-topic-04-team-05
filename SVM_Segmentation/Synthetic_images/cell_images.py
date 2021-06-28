import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from skimage.io import imread


def new_directory(base_dir_name, background_dir_name, object_dir_name, noise_dir_name):
    """
    creates new directory for background, noise and cells
    :return:
    """
    base_dir = str(base_dir_name)
    os.mkdir(base_dir)

    for element in [background_dir_name, object_dir_name, noise_dir_name]:
        #background_dir
        element = os.path.join(base_dir, str(element))
        os.mkdir(element)

#new_directory('base_dir', 'background_dir', 'cell_dir', 'noise_dir')

def show_image(path):
    """
    reads image from a path and shows it in a coordinate system
    :param path: path of the image
    :return: image in a coordinate system
    """
    image = imread(path)
    plt.imshow(image)
    plt.show()

#show_image('../../Data/N2DH-GOWT1/img/t01.tif')
image = imread('../../Data/N2DH-GOWT1/img/t01.tif')

def background_selection(image, directory, coordinate_list_background, coordinate_list_object, coordinate_list_noise):
    #Listen mit Koordinaten als Spalten nebeneinander
    coordinates = {'coordinate_list_background': coordinate_list_background,
            'coordinate_list_object': coordinate_list_object,
            'coordinate_list_noise': coordinate_list_noise}
    #name_list = ['y_coordinate', 'x_coordinate', 'heigth', 'width']
    dataframe = pd.DataFrame(coordinates)
    #dataframe_finished = dataframe.set_axis(name_list, axis=0)
    #Spalten auswählen
    for i in range(0,3):
        y_coordinate = dataframe.iloc[0, i]
        x_coordinate = dataframe.iloc[1, i]
        heigth = dataframe.iloc[2, i]
        width = dataframe.iloc[3, i]
        for directory_ in os.listdir(directory):
            object = str(directory_).split("_")[0]
            object = image[y_coordinate:y_coordinate+heigth, x_coordinate:x_coordinate+width].copy()
            plt.imshow(object)
            plt.show()
            path = (f'base_dir/{directory_}/1.tif')
            cv2.imwrite(path, object)

background_selection(image, 'base_dir', [0, 0, 200, 200], [50, 250, 150, 150], [200, 500, 50, 50])