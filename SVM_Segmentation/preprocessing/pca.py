import imageio as io
import imagecodecs
import numpy as np
import pandas as pd
import sklearn.decomposition as decomposition
from sklearn.preprocessing import StandardScaler
from SVM_Segmentation import read_images as rm
import matplotlib.pyplot as plt


def convert_pca(image, variance):
    """
    This function standardizes and scales the 2D-numpy-array, so that it can directly be used for finding its
    principal components. It then transforms it back to its original shape.
    :param variance: selected variance
    :param image: image that is supposed to be transformed
    :variance: a float between 0 and 1, which describes the variance that is supposed to be explained
    :return: an image, where the components explaining the most variance are kept
    """
    image = StandardScaler().fit_transform(image)
    pca = decomposition.PCA(variance)
    pca.fit(image)
    components = pca.transform(image)
    projected = pca.inverse_transform(components)
    return projected


def pca_different_images(image_path, principal_components):
    """
    The PCA is carried out for several different, but similiar images
    :param image_path: path of the images which are arranged into an dataframe and of which
    dimensions are reduced
    :param principal_components: the number of principal components that is supposed to kept, it can be maximum the
    number of images that are put in
    :return: returns a dataframe with several images, that are reduced in dimension
    """
    pca_list = []
    image_list = rm.read_image(image_path)
    image_names = rm.read_imagename(image_path)
    flattened = rm.image_flatten(image_list)
    df = rm.dataframe(flattened, image_names)
    arrays = df.values
    arrays_transposed = arrays.transpose()
    pca = decomposition.PCA(principal_components)
    pca.fit(arrays_transposed)
    components = pca.transform(arrays_transposed)
    return components


if __name__ == '__main__':
    image_read1 = io.imread('../../Data/N2DH-GOWT1/img/t01.png')
    pca1 = convert_pca(image_read1, 0.9)

    plt.imshow(pca1)
    plt.show()
