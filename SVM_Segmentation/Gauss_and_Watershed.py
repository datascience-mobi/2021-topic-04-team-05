import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import matplotlib.pyplot as plt

#Gaussian Filtering
image1 = cv2.imread('t01.tif')
GaussianFiltered = cv2.GaussianBlur(image1, (5,5),0)
#cv2.imshow('Original', image1)
#cv2.imshow('Gauss', GaussianFiltered)
#cv2.waitKey(0)
#cv2.destroyWindow()

#Watershed

# Load in image, convert to gray scale, and Otsu's threshold


img1 = cv2.imread("t01.tif")
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
thresh1 = cv2.dilate(thresh, kernel, iterations=9)

# the kernel matrix in this case (3,3) -> 3x3, and the iterations should be adapted in every image collection

# Compute Euclidean distance from every binary pixel to the nearest zero pixel then find peaks
distance_map = ndimage.distance_transform_edt(thresh)
local_max = peak_local_max(distance_map, indices=False, min_distance=60, labels=thresh1)
# Perform connected component analysis then apply Watershed
markers = ndimage.label(local_max, structure=np.ones((3,3)))[0]
labels = watershed(-distance_map, markers, mask=thresh1)

# Iterate through unique labels
total_area = 0
for label in np.unique(labels):
    if label == 0:
        continue

    # Create a mask
    mask = np.zeros(img.shape, dtype="uint8")
    mask[labels == label] = 255

    # Find contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.drawContours(img1, [max(cnts)], -1, (36,255,12), 4)

plt.imshow(img1)
plt.show()