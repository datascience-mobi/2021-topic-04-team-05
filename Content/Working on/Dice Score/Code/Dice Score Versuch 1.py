import numpy as np
import cv2

img = cv2.imread('plant.jpg', 0)
ret, th_groundtruth = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
ret, th_prediction = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

#dice score

dice = np.sum(th_prediction[th_groundtruth==th_prediction]*2/(np.sum(th_prediction)+np.sum(th_groundtruth)))
print(dice)