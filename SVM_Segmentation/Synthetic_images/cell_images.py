import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, color


def new_directory(base_dir_name, background_dir_name, object_dir_name, noise_dir_name, new_images_dir_name):
    """
    creates new directory for background, noise and objects
    :return:
    """
    base_dir = str(base_dir_name)
    os.mkdir(base_dir)
    new_images_dir = str(new_images_dir_name)
    os.mkdir(new_images_dir)

    for element in [background_dir_name, object_dir_name, noise_dir_name]:
        #background_dir
        element = os.path.join(base_dir, str(element))
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

def selection(image, directory, new_filename, coordinate_list_background, coordinate_list_object, coordinate_list_noise):
    #Listen mit Koordinaten als Spalten nebeneinander
    coordinates = {'coordinate_list_background': coordinate_list_background,
                  'coordinate_list_object': coordinate_list_object,
                  'coordinate_list_noise': coordinate_list_noise}
    # name_list = ['y_coordinate', 'x_coordinate', 'heigth', 'width']
    dataframe = pd.DataFrame(coordinates)
    # dataframe_finished = dataframe.set_axis(name_list, axis=0)
    # Spalten auswählen
    source = os.listdir(directory)
    for directory_ in source:
        y_coordinate = dataframe.iloc[0, source.index(str(directory_))]
        x_coordinate = dataframe.iloc[1, source.index(str(directory_))]
        heigth = dataframe.iloc[2, source.index(str(directory_))]
        width = dataframe.iloc[3, source.index(str(directory_))]
        object = image[y_coordinate:y_coordinate + heigth, x_coordinate:x_coordinate + width].copy()
        path = (f'base_dir/{directory_}/{new_filename}.tif')
        cv2.imwrite(path, object)
        #plt.imshow(object)
        #plt.show()

def resize_image(image_path, new_filename, heigth, width):
    background = io.imread(image_path)
    background_grey = color.rgb2gray(background)
    resized = cv2.resize(background_grey, (heigth, width))
    directory = str(image_path).rsplit('/',1)[0]
    path = (f'{directory}/{new_filename}.tif')
    cv2.imwrite(path, resized)

def generate_synthetic_images(background_path, object_path, new_image_path, new_filename, number_of_images,minimum_number_of_objects, maximum_number_of_objects, max_y, max_x):
    for i in range(0, number_of_images):
        # randomly choose the number of cells to put in the image
        num_cells_on_image = np.random.randint(minimum_number_of_objects, maximum_number_of_objects)

        # read the image
        background1 = cv2.imread(background_path)
        background1_grey = color.rgb2gray(background1)
        # add rotation to the background
        num_k = np.random.randint(0, 3)
        background_rotated1 = np.rot90(background1_grey, k=num_k)
        # resize the background to match what we want
        background_resized1 = np.matrix(cv2.resize(background_rotated1, (1024, 1024)))

        for i in range(0, num_cells_on_image):
            # read the image
            background2 = cv2.imread(background_path)
            background2_grey = color.rgb2gray(background2)
            # add rotation to the background
            num_k = np.random.randint(0, 3)
            background_rotated2 = np.rot90(background2_grey, k=num_k)
            # resize the background to match what we want
            background_resized2 = np.matrix(cv2.resize(background_rotated2, (1024, 1024)))

            # randomly choose a type of cell to add to the image
            object_version = np.random.randint(1, 3 + 1)

            readed_image = cv2.imread(f'{object_path}/{object_version}.tif')
            readed_image_grey = color.rgb2gray(readed_image)

            # add a random rotation to the cell
            rotated_image = np.rot90(readed_image_grey, k=np.random.randint(0, 3))

            # get a random x-coord
            y = np.random.randint(0, max_y)
            # get a random y-coord
            x = np.random.randint(0, max_x)
            # set the width and height
            h = rotated_image.shape[0]
            w = rotated_image.shape[1]

            # add the cell to the background
            background_resized1[y:y + h, x:x + w] = 0
            background_resized1[y:y + h, x:x + w] = rotated_image


    path = (f'{new_image_path}_{new_filename}_{i}.tif')
    cv2.imwrite(path, background_resized1)

if __name__ == '__main__':
    #new_directory('base_dir', 'background_dir', 'cell_dir', 'noise_dir', 'new_images_dir')

    #show_image('../../Data/N2DH-GOWT1/img/t01.tif')
    image = io.imread('../../Data/N2DH-GOWT1/img/t01.tif')

    selection(image, 'base_dir', 1, [0, 0, 200, 200], [750, 760, 100, 100], [200, 500, 50, 50])
    selection(image, 'base_dir', 2, [0, 0, 200, 200], [590, 380, 80, 80], [200, 500, 50, 50])
    selection(image, 'base_dir', 3, [0, 0, 200, 200], [330, 200, 60, 80], [200, 500, 50, 50])

    #resized_background = resize_image('base_dir/background_dir/1.tif', 'resized_background', 1024, 1024)

    #generate_synthetic_images('base_dir/background_dir/1.tif', 'base_dir/cell_dir', 'new_images_dir', 'generated_image', 10, 10, 31, 1400, 1000)

    background_path = 'base_dir/background_dir/1.tif'
    object_path = 'base_dir/cell_dir'
    new_image_path = 'new_images_dir'
    new_filename = 'generated_image'
    number_of_images = 10
    minimum_number_of_objects = 10
    maximum_number_of_objects = 31
    max_y = 1400
    max_x = 1000

    for i in range(0, number_of_images):
        # randomly choose the number of cells to put in the image
        num_objects_on_image = np.random.randint(minimum_number_of_objects, maximum_number_of_objects)

        #print(num_objects_on_image)

        # read the image
        background1 = io.imread(background_path)
        background1 = color.rgb2gray(background1)
        # add rotation to the background
        num_k = np.random.randint(0, 3)
        background1 = np.rot90(background1, k=num_k)
        # resize the background to match what we want
        background1 = np.matrix(cv2.resize(background1, (1024, 1024)))

        #print(background_resized1.shape)

        for number in range(0, num_objects_on_image):
            # read the image
            background2 = io.imread(background_path)
            background2 = color.rgb2gray(background2)
            # add rotation to the background
            num_k = np.random.randint(0, 3)
            background2 = np.rot90(background2, k=num_k)
            # resize the background to match what we want
            background2 = np.matrix(cv2.resize(background2, (1024, 1024)))

            #print(background_resized2)
            #print(background_resized2.shape)
            #plt.imshow(background_resized2)
            #plt.show()
            #print(background_resized1)
            #print(background_resized1.shape)
            #plt.imshow(background_resized1)
            #plt.show()

            # randomly choose a type of cell to add to the image
            object_version = np.random.randint(1, 3 + 1)

            object1 = io.imread(f'{object_path}/{object_version}.tif')
            object1 = color.rgb2gray(object1)
            #plt.imshow(readed_image_grey)
            #plt.show()

            # add a random rotation to the cell
            object1 = np.rot90(object1, k = np.random.randint(0, 3))

            print(object1.shape)


            object_shape = object1.shape
            # get a random x-coord
            y = np.random.randint(0, max_y)
            # get a random y-coord
            x = np.random.randint(0, max_x)
            # set the width and height
            h = (object1.shape)[0]
            w = (object1.shape)[1]

            print(h, w)
            #print(background2[y:y + h, x:x + w].shape)

            # add the cell to the background
            background2[y:y + h, x:x + w]  += object1

        background1 = np.minimum(background1, background2)

        #print(background_resized1.shape)
        #print(background_resized2.shape)
        #print(rotated_image)
        #print(rotated_image.shape)
        #plt.imshow(rotated_image)
        #plt.show()


        path = (f'{new_image_path}/{new_filename}_{i}.tif')
        cv2.imwrite(path, background1)