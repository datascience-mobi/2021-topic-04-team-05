import matplotlib.pyplot as plt
import cv2
import os


def gauss_filter(path):
    """
      This function filters images with a lot of noises
      :param: Path to folder
      :return: Images with Gauss Filter
      """
    original_image = cv2.imread(os.path.join(path))  # apply the gauss filter
    filtered_image = cv2.GaussianBlur(original_image, (5, 5), 0)
    return filtered_image

img_gauss = gauss_filter("../../Data/N2DH-GOWT1/img/t52.tif")
plt.imshow(img_gauss)
plt.show()