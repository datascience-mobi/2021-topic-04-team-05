import os
import cv2
import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, color


def new_directory(path, base_directory_name, background_directory_name, object_directory_name, object_directory_name_gt,
                  new_images_directory_name, new_images_directory_name_gt):
    """
    creates new directory for background and objects
    :param path: path were the new folders are supposed to be created
    :param base_directory_name: name of the base folder which is to be created
    :param background_directory_name: name of folder in base folder
    :param object_directory_name: name of another folder in base folder
    :param object_directory_name_gt: name of another folder in base folder
    :param new_images_directory_name: name of another folder in base folder
    :param new_images_directory_name_gt: name of another folder in base folder
    :return: a new directory with a folder structure where all with the main function created images are supposed
    to be saved
    """
    base_dir = os.path.join(str(path), base_directory_name)
    os.mkdir(base_dir)
    for element in [background_directory_name, object_directory_name, object_directory_name_gt,
                    new_images_directory_name, new_images_directory_name_gt]:
        element = os.path.join(base_dir, str(element))
        os.mkdir(element)


# noinspection PyShadowingNames
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


# noinspection PyShadowingNames,PyShadowingBuiltins
def selection(image: numpy.ndarray, gt: numpy.ndarray, directory_img_background, directory_img_object, directory_gt,
              new_filename, coordinate_list_background: list, coordinate_list_object: list):
    """
    this function cuts out objects of images
    :param image: image, of which objects are supposed to be cut out
    :param gt: ground truth, of which objects at the same coordinates as in the image are supposed to be cut out
    :param directory_img_background: directory, where the backgrounds are supposed to be saved
    :param directory_img_object: directory, where the objects of the image are supposed to be saved
    :param directory_gt: directory, where the objects of gt are supposed to be saved
    :param new_filename: the new filename, the cut out images should get
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
    source = [directory_img_background, directory_img_object]
    for directory_ in source:
        y_coordinate = dataframe.iloc[0, source.index(str(directory_))]
        x_coordinate = dataframe.iloc[1, source.index(str(directory_))]
        height = dataframe.iloc[2, source.index(str(directory_))]
        width = dataframe.iloc[3, source.index(str(directory_))]
        object = image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width].copy()
        path = f'{directory_}/{new_filename}.tif'
        cv2.imwrite(path, object)
    # gt objects
    y_coordinate = dataframe.iloc[0, 1]
    x_coordinate = dataframe.iloc[1, 1]
    height = dataframe.iloc[2, 1]
    width = dataframe.iloc[3, 1]
    object = gt[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width].copy()
    path = f'{directory_gt}/{new_filename}.tif'
    cv2.imwrite(path, object)


# noinspection PyShadowingNames
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
    directory = str(image_path).rsplit('/', 1)[0]
    path = f'{directory}/{new_filename}.tif'
    cv2.imwrite(path, resized)


# noinspection PyShadowingNames,DuplicatedCode
def generate_synthetic_images(background_path, object_path, gt_object_path, new_image_path, new_image_path_gt,
                              number_of_images: int, minimum_number_of_objects: int, maximum_number_of_objects: int,
                              size):
    """
    generates synthetic images
    :param background_path: path of the background
    :param object_path: path of the folder, where the objects are in
    :param gt_object_path: path of the folder, where the ground truth objects are in
    :param new_image_path: new folder path, where the new generated images are saved
    :param new_image_path_gt: new folder path, where the new generated gt images are saved
    :param number_of_images: the number of images to be generated
    :param minimum_number_of_objects: minimum number of objects that are pasted onto the background
    :param maximum_number_of_objects: maximum number of objects that are pasted onto the background
    :param size: size the generated picture is supposed to be
    :return: generated images
    """
    for i in range(0, number_of_images):
        # randomly choose number of cells to put in the image between defined minimum and maximum amount of objects
        num_objects_on_image = np.random.randint(minimum_number_of_objects, maximum_number_of_objects)

        # read image
        background1 = io.imread(background_path)
        background1 = color.rgb2gray(background1)

        # rotate backgrounds
        num_k = np.random.randint(0, 3)
        background1 = np.rot90(background1, k=num_k)

        gt_background1 = np.rot90(background1, k=num_k)

        # resize background
        background1 = np.matrix(cv2.resize(background1, size))

        gt_background1 = np.matrix(cv2.resize(gt_background1, size))
        gt_background1 = np.where(gt_background1 != 0, 0, gt_background1)

        for number in range(0, num_objects_on_image):
            # read image
            background2 = io.imread(background_path)
            background2 = color.rgb2gray(background2)

            # rotate background
            num_k = np.random.randint(0, 3)
            background2 = np.rot90(background2, k=num_k)

            gt_background2 = np.rot90(background2, k=num_k)

            # resize background
            background2 = np.matrix(cv2.resize(background2, size))

            gt_background2 = np.matrix(cv2.resize(gt_background2, size))

            gt_background2 = np.where(gt_background2 != 0, 0, gt_background2)

            # randomly choose a type of object
            object_version = np.random.randint(1, 3 + 1)

            object1 = io.imread(f'{object_path}/{object_version}.tif')
            object1 = color.rgb2gray(object1)

            gt_object1 = io.imread(f'{gt_object_path}/{object_version}.tif')
            gt_object1 = color.rgb2gray(gt_object1)

            # random rotation of the object
            k = np.random.randint(0, 3)
            object1 = np.rot90(object1, k)

            gt_object1 = np.rot90(gt_object1, k)

            # set the width and height
            h = object1.shape[0]
            w = object1.shape[1]

            # get a random x-coordinate
            max_y_new = background1.shape[0] - h
            y = np.random.randint(0, max_y_new)
            # get a random y-coordinate
            max_x_new = background1.shape[1] - w
            x = np.random.randint(0, max_x_new)

            # add the cell to the background
            background2[y:y + h, x:x + w] = 0
            background2[y:y + h, x:x + w] = object1

            gt_background2[y:y + h, x:x + w] += gt_object1

            # choose maximum value for each pixel, as the cells have a higher intensity than the background and all
            background1 = np.maximum(background1, background2)

            gt_background1 = np.maximum(gt_background1, gt_background2)

        path = f'{new_image_path}/{i}.tif'
        cv2.imwrite(path, background1)

        path = f'{new_image_path_gt}/{i}.tif'
        cv2.imwrite(path, gt_background1)


# noinspection PyShadowingNames
def main(image_path, gt_path, name, background_coord_list, object1_coord_list, object2_coord_list, object3_coord_list,
         number_of_images, minimum_number_of_objects, maximum_number_of_objects):
    """
    executes previous functions in order to generate synthetic images and their according ground truth and save them
    :param image_path: path of the image where the objects and background for the generated image are
    supposed to be cut out
    :param gt_path: path of the ground truth image where the objects for the generated ground truth are
    supposed to be cut out
    :param name: name of the folder where the cut and generated images are supposed to be saved
    :param background_coord_list: the coordinates of the part of the image, which is supposed to be the background
    of the generated image in the following format: [y_coordinate, x_coordinate, height, width]
    :param object1_coord_list: the coordinates of the part of the image, which is supposed to be the object 1
    of the generated image in the following format: [y_coordinate, x_coordinate, height, width]
    :param object2_coord_list: the coordinates of the part of the image, which is supposed to be the object 2
    of the generated image in the following format: [y_coordinate, x_coordinate, height, width]
    :param object3_coord_list: the coordinates of the part of the image, which is supposed to be the object 3
    of the generated image in the following format: [y_coordinate, x_coordinate, height, width]
    :param number_of_images: the number of images that is supposed to be generated
    :param minimum_number_of_objects: the minimum number of objects that are supposed to be pasted into the
    generated image
    :param maximum_number_of_objects: the maximum number of objects that are supposed to be pasted into the
    generated image
    :return: generated images in the according folders
    """
    image = io.imread(image_path)
    gt = io.imread(gt_path)
    gt_transposed = gt.transpose()
    size = gt_transposed.shape
    directory_img_background = f'..\\..\\Data\\synthetic_cell_images\\{name}\\background'
    directory_img_object = f'..\\..\\Data\\synthetic_cell_images\\{name}\\cell'
    directory_gf = f'..\\..\\Data\\synthetic_cell_images\\{name}\\cell_gt'
    selection(image, gt, directory_img_background, directory_img_object, directory_gf, 1, background_coord_list,
              object1_coord_list)
    selection(image, gt, directory_img_background, directory_img_object, directory_gf, 2, background_coord_list,
              object2_coord_list)
    selection(image, gt, directory_img_background, directory_img_object, directory_gf, 3, background_coord_list,
              object3_coord_list)

    background_path = f'../../Data/synthetic_cell_images/{name}/background/1.tif'
    object_path = f'../../Data/synthetic_cell_images/{name}/cell'
    gt_object_path = f'../../Data/synthetic_cell_images/{name}/cell_gt'
    new_image_path = f'../../Data/synthetic_cell_images/{name}/generated_images_img'
    new_image_path_gt = f'../../Data/synthetic_cell_images/{name}/generated_images_gt'

    generate_synthetic_images(background_path, object_path, gt_object_path, new_image_path, new_image_path_gt,
                              number_of_images, minimum_number_of_objects, maximum_number_of_objects, size)


if __name__ == '__main__':
    directory = '..\\..\\Data\\synthetic_cell_images\\'
    dirs = ['N2DH-GOWT1_t01', 'N2DL-HeLa_t13', 'NIH3T3_dna-0']
    for new_dir in dirs:
        if os.path.exists(os.path.join(f'{directory}{new_dir}')) is False:
            new_directory(directory, new_dir, 'background', 'cell', 'cell_gt', 'generated_images_img',
                          'generated_images_gt')

    # N2DH-GOWT1

    image_path = '..\\..\\Data\\N2DH-GOWT1\\img\\t01.tif'
    gt_path = '..\\..\\Data\\N2DH-GOWT1\\gt\\tif\\man_seg01.tif'
    background_coord_list = [0, 0, 200, 200]
    object1_coord_list = [750, 760, 100, 100]
    object2_coord_list = [590, 380, 80, 80]
    object3_coord_list = [330, 200, 60, 80]
    name = 'N2DH-GOWT1_t01'
    number_of_images = 10
    # between 10 and 15 cells
    min_objects = 10
    max_objects = 15
    main(image_path, gt_path, name, background_coord_list, object1_coord_list, object2_coord_list, object3_coord_list,
         number_of_images, min_objects, max_objects)

    # N2DL-HeLa

    image_path = '../../Data/N2DL-HeLa/img/t13.tif'
    gt_path = '../../Data/N2DL-HeLa/gt/man_seg13.tif'
    background_coord_list = [0, 800, 170, 190]
    object1_coord_list = [160, 450, 40, 40]
    object2_coord_list = [330, 772, 40, 35]
    object3_coord_list = [430, 60, 40, 50]
    name = 'N2DL-HeLa_t13'
    number_of_images = 10
    # between 80 and 120 cells
    min_objects = 80
    max_objects = 120
    main(image_path, gt_path, name, background_coord_list, object1_coord_list, object2_coord_list, object3_coord_list,
         number_of_images, min_objects, max_objects)

    # NIH3T3

    image_path = '../../Data/NIH3T3/img/dna-0.png'
    gt_path = '../../Data/NIH3T3/gt/0.png'
    background_coord_list = [550, 1200, 100, 100]
    object1_coord_list = [105, 235, 100, 100]
    object2_coord_list = [420, 1150, 130, 120]
    object3_coord_list = [770, 50, 110, 100]
    name = 'NIH3T3_dna-0'
    number_of_images = 10
    # between 30 and 40 cells
    min_objects = 30
    max_objects = 40
    main(image_path, gt_path, name, background_coord_list, object1_coord_list, object2_coord_list, object3_coord_list,
         number_of_images, min_objects, max_objects)

    # showing one generated image of each set and its according ground truth

    image = io.imread('../../Data/synthetic_cell_images/NIH3T3_dna-0/generated_images_gt/1.tif')
    image_gen = io.imread('../../Data/synthetic_cell_images/NIH3T3_dna-0/generated_images_img/1.tif')
    image1 = io.imread('../../Data/synthetic_cell_images/N2DL-HeLa_t13/generated_images_gt/1.tif')
    image_gen1 = io.imread('../../Data/synthetic_cell_images/N2DL-HeLa_t13/generated_images_img/1.tif')
    image2 = io.imread('../../Data/synthetic_cell_images/N2DH-GOWT1_t01/generated_images_gt/1.tif')
    image_gen2 = io.imread('../../Data/synthetic_cell_images/N2DH-GOWT1_t01/generated_images_img/1.tif')
    plt.imshow(image)
    plt.show()
    plt.imshow(image_gen)
    plt.show()
    plt.imshow(image1)
    plt.show()
    plt.imshow(image_gen1)
    plt.show()
    plt.imshow(image2)
    plt.show()
    plt.imshow(image_gen2)
    plt.show()
