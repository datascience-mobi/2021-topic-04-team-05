import cv2
import readimages as rm
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.morphology import dilation
from skimage import measure

from skimage import data
from skimage.color import rgb2gray
from skimage.morphology import dilation
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max



# Load in image, convert to gray scale, and Otsu's threshold

#def watershed(path_of_imagefolder):
 #   image_list = rm.read_image(path_of_imagefolder)
  #  watershed_image_list = []
   # for image in image_list:
    #    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #erstezen durch skimage/matplotlib funktion
     #   ret1, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # ersetzen durch skimage/matplotlib funktion
      #  kernel = np.ones((3, 3), np.uint8)
       # dilated = dilation(thresholded, kernel, iterations=8)  # ersetzen durch skimage/matplotlib funktion
        ## the kernel matrix in this case (3,3) -> 3x3, and the iterations should be adapted in every image collection
       # distance_map = ndimage.distance_transform_edt(thresholded)
        # Compute Euclidean distance from every binary pixel to the nearest zero pixel then find peaks
       # local_max = peak_local_max(distance_map, indices=False, min_distance=60, labels=dilated)
       # # Perform connected component analysis then apply Watershed
      #  markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
       # labels = watershed(-distance_map, markers, mask=dilated)
        # Iterate through unique labels
 #       total_area = 0
  #      for label in np.unique(labels):
  #          if label == 0:
   #             continue
    #        # Create a mask
     #       mask = np.zeros(gray.shape, dtype="uint8")
    #        mask[labels == label] = 255
            # Find contours
      #      contours = measure.find_contours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # ersetzen durch skimage/matplotlib funktion
       #     contours = contours[0] if len(contours) == 2 else contours[1]
        #    c = max(contours, key=cv2.contourArea)  # ersetzen durch skimage/matplotlib funktion
         #   cv2.drawContours(image, [c], -1, (36, 255, 12), 4)  # ersetzen durch skimage/matplotlib funktion
          #  watershed_image_list.append(image)



#watershed('../Data/N2DH-GOWT1/img')

#plt.imshow(watershed('../Data/N2DH-GOWT1/img'))
#plt.show()

#def watershed(path_of_imagefolder):
 #   image_list = rm.read_image(path_of_imagefolder)
  #  watershed_image_list = []


def watershed(path_of_image_folder):
    image_list = rm.read_image(path_of_image_folder)
    watershed_image_list = []
    for image in watershed_image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # erstezen durch skimage/matplotlib funktion
        ret1, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # ersetzen durch skimage/matplotlib funktion
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresholded, kernel, iterations=9)  # ersetzen durch skimage/matplotlib funktion
    # the kernel matrix in this case (3,3) -> 3x3, and the iterations should be adapted in every image collection
        distance_map = ndimage.distance_transform_edt(thresholded)
    # Compute Euclidean distance from every binary pixel to the nearest zero pixel then find peaks
        local_max = peak_local_max(distance_map, indices=False, min_distance=60, labels=dilated)
    # Perform connected component analysis then apply Watershed
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        labels = cv2.watershed(-distance_map, markers)
    # Iterate through unique labels
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


#if __name__ == '__main__':
 #   watersheded_images = watershed('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/gt/jpg')

    #for i in range(1, 6):
     #   img = watersheded_images
     #   if img is not None:
     #       img = float(img)
 #   plt.imshow(img)
  #  plt.show()

#cv2.imshow('Trial', watersheded_images)[0]
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(dtype(img))