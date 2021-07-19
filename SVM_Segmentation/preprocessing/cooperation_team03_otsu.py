import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def otsu(image, intensity_lvls=256):
    """
    This function takes an image and calculates the probability of class occurrence and the mean value for all pixels to
     calculate the threshold according to the formula of Otsu Thresholding without using a for loop.
     Also it calculates the total variance and uses it to calculate the goodness of the threshold.
    :param image: Input image
    :param intensity_lvls:The total number of intensity levels
    :return: Threshold and goodness of the image
    """

    histogram = np.histogram(image, bins=np.arange(intensity_lvls + 1), density=True)

    class_probability = np.cumsum(histogram[0])
    class_mean = np.cumsum(histogram[0] * np.arange(intensity_lvls))
    total_mean = np.mean(image)

    with np.errstate(divide='ignore'):
        inbetween_variance = (total_mean * class_probability - class_mean) ** 2 / (
                class_probability * (1 - class_probability))

    # Inf values are invalid
    inbetween_variance[inbetween_variance == np.inf] = np.nan
    optimal_threshold = np.nanargmax(inbetween_variance)

    return optimal_threshold


def clipping(img, threshold):
    """
    This function takes the intensity of every pixel and sets its value to 0 if the threshold is equal or smaller 0.
    If the intensity value is greater than the threshold, the value is set to 1.
    :param img: Input image
    :param threshold: Threshold that defines the clipping
    :return: Clipped image
    """

    workimg = np.zeros(img.shape)
    workimg[img > threshold] = 1

    return workimg


def complete_segmentation(img, intensity_lvls=256):
    """
    Performs complete image segmentation using Otsu threshold.
    :param intensity_lvls: Number of intensity values
    :param img: Image to be segmented
    :return: Segmented binary image
    """
    threshold = otsu(img, intensity_lvls)
    workimg = clipping(img, threshold)

    return workimg


if __name__ == '__main__':
    image = io.imread('../../Data/N2DH-GOWT1/img/t21.tif')
    segmented_image = complete_segmentation(image, intensity_lvls=2**16)
    plt.imshow(segmented_image)
    plt.show()
