import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def watershed(path_to_folder):
    images = []
    for filename in os.listdir(path_to_folder):
        original_image = cv2.imread(os.path.join(path_to_folder,filename))
        # Segmentation
        grayconverted = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(grayconverted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Further noise removal, the kernel matrix in this case (3,3) -> 3x3
        kernel = np.ones((4, 4), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # clear background area, dilate the matrix
        clear_background = cv2.dilate(opening, kernel, iterations=3)
        # Finding clear foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, clear_foreground = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
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
        original_image[markers == -1] = [255, 0, 20]

        if original_image is not None:
            images.append(original_image)
    return images

if __name__ == '__main__':
    path = ("/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img")
    max = os.listdir(path)
    for i in range (1, len(max)):
        segmented_images = watershed(path)
        plt.imshow(segmented_images[i])
        plt.show()


















