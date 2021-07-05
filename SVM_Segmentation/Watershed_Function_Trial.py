import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import matplotlib.pyplot as plt

# Load in image, convert to gray scale, and Otsu's threshold

def watershed(i):
    img1 = cv2.imread(i)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret1, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=8)
    # the kernel matrix in this case (3,3) -> 3x3, and the iterations should be adapted in every image collection
    distance_map = ndimage.distance_transform_edt(thresholded)
    # Compute Euclidean distance from every binary pixel to the nearest zero pixel then find peaks
    local_max = peak_local_max(distance_map, indices=False, min_distance=60, labels=dilated)
    # Perform connected component analysis then apply Watershed
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance_map, markers, mask=dilated)
    # Iterate through unique labels
    total_area = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        # Create a mask
        mask = np.zeros(img.shape, dtype="uint8")
        mask[labels == label] = 255
        # Find contours
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(img1, [c], -1, (36, 255, 12), 4)

    plt.imshow(img1)
    plt.show()


watershed('t01.tif')