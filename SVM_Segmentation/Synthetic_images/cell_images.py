import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from skimage.io import imread


def new_directory(base_dir_name, background_dir_name, object_dir_name, noise_dir_name, new_images_dir_name):
    """
    creates new directory for background, noise and cells
    :return:
    """
    base_dir = str(base_dir_name)
    os.mkdir(base_dir)
    new_images_dir = str(new_images_dir_name)

    for element in [background_dir_name, object_dir_name, noise_dir_name]:
        #background_dir
        element = os.path.join(base_dir, str(element))
        os.mkdir(element)

new_directory('base_dir', 'background_dir', 'cell_dir', 'noise_dir', 'new_images_dir')

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

def selection(image, directory, new_filename, coordinate_list_background, coordinate_list_object, coordinate_list_noise):
    #Listen mit Koordinaten als Spalten nebeneinander
    coordinates = {'coordinate_list_background': coordinate_list_background,
            'coordinate_list_object': coordinate_list_object,
            'coordinate_list_noise': coordinate_list_noise}
    #name_list = ['y_coordinate', 'x_coordinate', 'heigth', 'width']
    dataframe = pd.DataFrame(coordinates)
    #dataframe_finished = dataframe.set_axis(name_list, axis=0)
    #Spalten auswählen
    for directory_ in os.listdir(directory):
        for i in range(0, 3):
            y_coordinate = dataframe.iloc[0, i]
            x_coordinate = dataframe.iloc[1, i]
            heigth = dataframe.iloc[2, i]
            width = dataframe.iloc[3, i]
            object = str(directory_).split("_")[0]
            object = image[y_coordinate:y_coordinate+heigth, x_coordinate:x_coordinate+width].copy()
            path = (f'base_dir/{directory_}/{new_filename}.tif')
            cv2.imwrite(path, object)
            plt.imshow(object)
            plt.show()

selection(image, 'base_dir', 1, [0, 0, 200, 200], [750, 760, 100, 100], [200, 500, 50, 50])

selection(image, 'base_dir', 2, [0, 0, 200, 200], [590, 380, 80, 80], [200, 500, 50, 50])

selection(image, 'base_dir', 3, [0, 0, 200, 200], [330, 200, 60, 80], [200, 500, 50, 50])

def resize_image(image_path, new_filename, heigth, width):
    background = imread(image_path)
    resized = cv2.resize(background, (heigth, width))
    directory = str(image_path).rsplit('/',1)[0]
    path = (f'{directory}/{new_filename}.tif')
    cv2.imwrite(path, resized)

resized_background = resize_image('base_dir/background_dir/1.tif', 'resized_background', 1024, 1024)


num_images_wanted = 10
min_cells_on_image = 1
max_cells_on_image = 100
#set max x and y to prevent cells from extending outside the bachground image i.e.
#if the get place too close to the edge.
max_x = 1400
max_y = 1000

for i in range(0, num_images_wanted):
    # randomly choose the number of cells to put in the image
    num_cells_on_image = np.random.randint(min_cells_on_image, max_cells_on_image + 1)

    # Name the image.
    # The number of cells is included in the file name.
    image_name = 'image_' + str(i) + '_' + str(num_cells_on_image) + '_.png'

    # =========================
    # 1. Create the background
    # =========================

    path = 'base_dir/bground_dir/bground_1.png'

    # read the image
    bground_comb = cv2.imread(path)

    # add random rotation to the background
    num_k = np.random.randint(0, 3)
    bground_comb = np.rot90(bground_comb, k=num_k)

    # resize the background to match what we want
    bground_comb = cv2.resize(bground_comb, (1600, 1200))

    # ===============================
    # 2. Add cells to the background
    # ===============================

    for j in range(0, num_cells_on_image):

        path = 'base_dir/bground_dir/bground_1.png'

        # read the image
        bground = cv2.imread(path)
        # add rotation to the background
        bground = np.rot90(bground, k=num_k)
        # resize the background to match what we want
        bground = cv2.resize(bground, (1600, 1200))

        # randomly choose a type of cell to add to the image
        cell_type = np.random.randint(1, 3 + 1)

        if cell_type == 1:
            # cell_1 path
            cell_1 = cv2.imread('base_dir/cell_dir/cell_1.png')

            # add a random rotation to the cell
            cell_1 = np.rot90(cell_1, k=np.random.randint(0, 3))

            # get the shape after rotation
            shape = cell_1.shape

            # get a random x-coord
            y = np.random.randint(0, max_y)
            # get a random y-coord
            x = np.random.randint(0, max_x)
            # set the width and height
            h = shape[0]
            w = shape[1]

            # add the cell to the background
            bground[y:y + h, x:x + w] = 0
            bground[y:y + h, x:x + w] = cell_1

        if cell_type == 2:
            cell_2 = cv2.imread('base_dir/cell_dir/cell_2.png')

            # add a random rotation to the cell
            cell_2 = np.rot90(cell_2, k=np.random.randint(0, 3))

            shape = cell_2.shape

            y = np.random.randint(0, max_y)
            x = np.random.randint(0, max_x)
            h = shape[0]
            w = shape[1]

            bground[y:y + h, x:x + w] = 0
            bground[y:y + h, x:x + w] = cell_2

        if cell_type == 3:
            # cell_3

            cell_3 = cv2.imread('base_dir/cell_dir/cell_3.png')

            # add a random rotation to the cell
            cell_3 = np.rot90(cell_3, k=np.random.randint(0, 3))

            shape = cell_3.shape

            y = np.random.randint(0, max_y)
            x = np.random.randint(0, max_x)
            h = shape[0]
            w = shape[1]

            bground[y:y + h, x:x + w] = 0
            bground[y:y + h, x:x + w] = cell_3

        bground_comb = np.minimum(bground_comb, bground)

        # =============================================
        # 3. Add noise and artifacts to the background
        # =============================================

        # We will only add 3 noise items to each image
        for k in range(0, 3):

            path = 'base_dir/bground_dir/bground_1.png'

            # read the image
            bground = cv2.imread(path)
            # add rotation to the background
            bground = np.rot90(bground, k=num_k)
            # resize the background to match what we want
            bground = cv2.resize(bground, (1600, 1200))

            # randomly choose a type of cell to add to the image
            noise_type = np.random.randint(1, 3 + 1)

            if noise_type == 1:
                # cell_1 path
                noise_1 = cv2.imread('base_dir/noise_dir/noise_1.png')

                # add a random rotation to the cell
                noise_1 = np.rot90(noise_1, k=np.random.randint(0, 3))

                # get the shape after rotation
                shape = noise_1.shape

                # get a random x-coord
                y = np.random.randint(0, max_y)
                # get a random y-coord
                x = np.random.randint(0, max_x)
                # set the width and height
                h = shape[0]
                w = shape[1]

                # add the cell to the background
                bground[y:y + h, x:x + w] = 0
                bground[y:y + h, x:x + w] = noise_1

            if noise_type == 2:
                noise_2 = cv2.imread('base_dir/noise_dir/noise_2.png')

                # add a random rotation to the cell
                noise_2 = np.rot90(noise_2, k=np.random.randint(0, 3))

                shape = noise_2.shape

                y = np.random.randint(0, max_y)
                x = np.random.randint(0, max_x)
                h = shape[0]
                w = shape[1]

                bground[y:y + h, x:x + w] = 0
                bground[y:y + h, x:x + w] = noise_2

            if noise_type == 3:
                # noise_3

                noise_3 = cv2.imread('base_dir/noise_dir/noise_3.png')

                # add a random rotation to the cell
                noise_3 = np.rot90(noise_3, k=np.random.randint(0, 3))

                shape = noise_3.shape

                y = np.random.randint(0, max_y)
                x = np.random.randint(0, max_x)
                h = shape[0]
                w = shape[1]

                bground[y:y + h, x:x + w] = 0
                bground[y:y + h, x:x + w] = noise_3

            bground_comb = np.minimum(bground_comb, bground)

            # ===============================
            # 3. Save the image
            # ===============================

            path = 'new_images_dir/' + image_name
            cv2.imwrite(path, bground_comb)

        print('Num imgaes created: ', num_images_wanted)




