import cv2
import numpy as np
from skimage import color


def watershed(image):
    """
       This is a gradient-ascend-based super pixel algorithm function - a "rough" segmentation. It also can be use for
       comparison with the SVM.
       :param: Path to folder to extract each image
       :return: With watershed segmented images
       """
    original_image = cv2.imread(image)
    # Segmentation through threshold
    gray_converted = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray_converted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Further noise removal from threshold image, the kernel matrix in this case (3,3) -> 3x3
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    # clear background area by matrix dilation
    clear_background = cv2.dilate(opening, kernel, iterations=3)
    # Finding clear foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, clear_foreground = cv2.threshold(dist_transform, 0.001 * dist_transform.max(), 255, 0)
    # Finding unknown region
    clear_foreground = np.uint8(clear_foreground)
    unknown = cv2.subtract(clear_background, clear_foreground)
    # Marker labelling
    ret, markers = cv2.connectedComponents(clear_foreground)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(original_image, markers)
    original_image[markers == -1] = [255, 0, 0]
    gray = color.rgb2gray(original_image)

    return gray


if __name__ == '__main__':
    ws = watershed("/Users/laurasanchis/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img/t72.png")
    print(ws)

