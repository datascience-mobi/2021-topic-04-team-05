import os
import cv2
import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, color


def new_directory(path, base_directory_name, background_directory_name, object_directory_name, new_images_directory_name):
    """
    creates new directory for background and objects
    :return:
    """
    base_dir = os.path.join(str(path), base_directory_name)
    os.mkdir(base_dir)
    new_images_dir = os.path.join(str(path), new_images_directory_name)
    os.mkdir(new_images_dir)
    for element in [background_directory_name, object_directory_name]:
        #background_dir
        element = os.path.join(str(path), base_dir, str(element))
        os.mkdir(element)


def show_image(path):
    """
    reads image from a path and shows it in a coordinate system
    :param path: path of the image
    :return: image in a coordinate system
    """
    image = io.imread(path)
    image_grey = color.rgb2gray(image)
    plt.imshow(image_grey)
    plt.show()

def selection(image: numpy.ndarray, directory, new_filename, coordinate_list_background: list, coordinate_list_object: list):
    """
    objects can be cut out of images with this function
    :param image: image, of which objects are supposed to be cut out
    :param directory: directory, where the objects are supposed to be saved
    :param new_filename:
    :param coordinate_list_background: the coordinates of the part of the image, which is supposed to be the background
    of the generated image in the following format: [y_coordinate, x_coordinate, height, width]
    :param coordinate_list_object: the coordinates of the part of the image, which is supposed to be the object
    of the generated image in the following format: [y_coordinate, x_coordinate, height, width]
    :return: cut images, saved in the according directories in the given directory
    """
    # create lists of coordinates as columns next to each other
    coordinates = {'coordinate_list_background': coordinate_list_background,
                  'coordinate_list_object': coordinate_list_object}
    dataframe = pd.DataFrame(coordinates)
    # choose columns
    source = os.listdir(directory)
    for directory_ in source:
        y_coordinate = dataframe.iloc[0, source.index(str(directory_))]
        x_coordinate = dataframe.iloc[1, source.index(str(directory_))]
        height = dataframe.iloc[2, source.index(str(directory_))]
        width = dataframe.iloc[3, source.index(str(directory_))]
        object = image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width].copy()
        path = (f'{directory}/{directory_}/{new_filename}.tif')
        cv2.imwrite(path, object)

def resize_image(image_path, new_filename, height, width):
    """
    resizes image
    :param image_path: path of the image to be resized
    :param new_filename: new filename of resized image
    :param height: wanted height of the new image
    :param width: wanted width of the new image
    :return: resized image
    """
    background = io.imread(image_path)
    background_grey = color.rgb2gray(background)
    resized = cv2.resize(background_grey, (height, width))
    directory = str(image_path).rsplit('/',1)[0]
    path = (f'{directory}/{new_filename}.tif')
    cv2.imwrite(path, resized)

def generate_synthetic_images(background_path, object_path, new_image_path, new_filename: str, number_of_images: int,minimum_number_of_objects: int, maximum_number_of_objects: int):
    """
    generates synthetic images
    :param background_path: path of the background
    :param object_path: path of the folder, where the objects are in
    :param new_image_path: new folder path, where the new generated images are saved
    :param new_filename: a string describing the generated images, which will be included in the new filename of the generated images
    :param number_of_images: the number of images to be generated
    :param minimum_number_of_objects: minimum number of objects that are pasted onto the background
    :param maximum_number_of_objects: maximum number of objects that are pasted onto the background
    :return: generated images
    """
    for i in range(0, number_of_images):
        # randomly choose the number of cells to put in the image
        num_objects_on_image = np.random.randint(minimum_number_of_objects, maximum_number_of_objects)

        # read the image
        background1 = io.imread(background_path)
        background1 = color.rgb2gray(background1)
        # add rotation to the background
        num_k = np.random.randint(0, 3)
        background1 = np.rot90(background1, k=num_k)
        # resize the background to match what we want
        background1 = np.matrix(cv2.resize(background1, (1024, 1024)))

        list = []
        for number in range(0, num_objects_on_image):
            # read the image
            background2 = io.imread(background_path)
            background2 = color.rgb2gray(background2)
            # add rotation to the background
            num_k = np.random.randint(0, 3)
            background2 = np.rot90(background2, k=num_k)
            # resize the background to match what we want
            background2 = np.matrix(cv2.resize(background2, (1024, 1024)))

            # randomly choose a type of cell to add to the image
            object_version = np.random.randint(1, 3 + 1)

            object1 = io.imread(f'{object_path}/{object_version}.tif')
            object1 = color.rgb2gray(object1)

            # add a random rotation to the cell
            object1 = np.rot90(object1, k=np.random.randint(0, 3))

            object_shape = object1.shape

            # set the width and height
            h = (object1.shape)[0]
            w = (object1.shape)[1]

            # get a random x-coordinate
            max_y_new = background1.shape[0] - h
            y = np.random.randint(0, max_y_new)
            # get a random y-coord
            max_x_new = background1.shape[1] - w
            x = np.random.randint(0, max_x_new)

            # add the cell to the background
            background2[y:y + h, x:x + w] += object1

            # choose maximum value for each pixel, as the cells have a higher intensity than the background and all
            # created images have to be fused in one
            background1 = np.maximum(background1, background2)

        path = (f'{new_image_path}/{new_filename}_{i}.tif')
        cv2.imwrite(path, background1)

if __name__ == '__main__':
    directory = '..\\..\\Data\\synthetic_images\\'
    new_dir = 'N2DH-GOWT1_t01'
    if os.path.exists(os.path.join(str(directory), str(new_directory))) is True:
        new_directory(directory, new_dir, 'background', 'cell', 'generated_images')

    image = io.imread('..\\..\\Data\\N2DH-GOWT1\\img\\t01.tif')
    selection(image, '..\\..\\Data\\synthetic_images\\N2DH-GOWT1_t01\\', 1, [0, 0, 200, 200], [750, 760, 100, 100])
    selection(image, '..\\..\\Data\\synthetic_images\\N2DH-GOWT1_t01\\', 2, [0, 0, 200, 200], [590, 380, 80, 80])
    selection(image, '..\\..\\Data\\synthetic_images\\N2DH-GOWT1_t01\\', 3, [0, 0, 200, 200], [330, 200, 60, 80])

    background_path = '../../Data/synthetic_images/N2DH-GOWT1_t01/background/1.tif'
    object_path = '../../Data/synthetic_images/N2DH-GOWT1_t01/cell'
    new_image_path = '../../Data/synthetic_images/generated_images'
    new_filename = 'generated_image'

    generate_synthetic_images(background_path, object_path, new_image_path, new_filename, 10, 10, 20)

    image_gen = io.imread('../../Data/synthetic_images/generated_images/generated_image_2.tif')
    plt.imshow(image_gen)
    plt.show()