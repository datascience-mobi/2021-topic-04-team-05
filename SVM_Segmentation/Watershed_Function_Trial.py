import cv2
import readimages as rm
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import dilation
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max



def watershed(path_of_image_folder):
    image_list = rm.read_image(path_of_image_folder)
    watershed_image_list = []
    for image in watershed_image_list:
        gray = rgb2gray(image)  # erstezen durch skimage/matplotlib funktion
        ret1, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # ersetzen durch skimage/matplotlib funktion
        kernel = np.ones((3, 3), np.uint8)
        dilated = dilation(thresholded, kernel, iterations=9)  # ersetzen durch skimage/matplotlib funktion
    # the kernel matrix in this case (3,3) -> 3x3, and the iterations should be adapted in every image collection
        distance_map = ndimage.distance_transform_edt(thresholded)
    # Compute Euclidean distance from every binary pixel to the nearest zero pixel then find peaks
        local_max = peak_local_max(distance_map, indices=False, min_distance=60, labels=dilated)
    # Perform connected component analysis then apply Watershed
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance_map, markers, mask=dilated)
    # Iterate through unique labels
        total_area = 0
        unique_elements = np.unique(labels)
        for label in unique_elements:
            if label == 0:
                continue

            # Create a mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # Find contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.drawContours(gray, [max(cnts)], -1, (36, 255, 12), 4)
        watershed_image_list.append(image)

    return image_list

if __name__ == '__main__':
    for i in range(1,6):
        a = watershed('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img')
        plt.imshow(a[i])
        plt.show()



   # for i in range(1, 6):
     #   img = watersheded_images
     #   if img is not None:
     #       img = float(img)
 #   plt.imshow(img)
  #  plt.show()

#cv2.imshow('Trial', watersheded_images)[0]
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(dtype(img))