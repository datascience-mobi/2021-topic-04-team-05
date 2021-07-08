import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from skimage.transform import resize
from skimage import io
import sklearn.decomposition as skdecomp
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import cv2

def oneD_array_to_twoD_array(oneDarray):
    twoDarray = np.stack(oneDarray, axis=0)
    a = int(math.sqrt(len(twoDarray)))
    twoDarray = twoDarray.reshape(a, a)
    return twoDarray

def pca(image, variance):
    image = StandardScaler().fit_transform(image)
    pca = skdecomp.PCA(variance)
    pca.fit(image)
    components = pca.transform(image)
    projected = pca.inverse_transform(components)
    if projected is not None:
        return projected

imgread = io.imread('../Data/test/img/t01.tif')
#imgtile = imgread[0:, 550:551]
imgpca = pca(imgread, 0.75)
imgresize = resize(imgpca, (50, 50))

#plt.imshow(imgresize)
#plt.show()

imgflat = imgresize.flatten()
imgnormal = np.asarray(imgflat).transpose()

X = pd.DataFrame(data=imgnormal)
X.insert(loc=len(X.columns), column='intercept', value=1)

# read in ground truth images
gtread = io.imread('../Data/test/gt/man_seg01.jpg')
#gttile = gtread[0:, 550:551]

# thresholding ground truth images to get black-and-white-only images
gtresize = resize(gtread, (50, 50))

#plt.imshow(gtresize)
#plt.show()

gtthreshold = (cv2.threshold(gtresize, 0, 1, cv2.THRESH_BINARY))[1]
y_labels = np.where(gtthreshold == 0, -1, gtthreshold)

#plt.imshow(gtthreshold)
#plt.show()

#plt.imshow(y_labels)
#plt.show()

gtflat = gtthreshold.flatten()

# Turning gt values into 1 and -1 labels
y_labels_flat = np.where(gtflat == 0, -1, gtflat)
Y = pd.DataFrame(data=y_labels_flat)

segmented_image = oneD_array_to_twoD_array(y_labels_flat)
#plt.imshow(segmented_image)
#plt.show()

y_train_predicted = np.array([1,1,-1,-1])
y_test_predicted = np.array([[1,1,-1,-1]])
y_svm = np.array([])
y_svm_train = np.append(y_svm, y_train_predicted)
y_svm_test = np.append(y_svm_train, y_test_predicted)

a = np.array([[1,2],[1,2],[1,2]])
b = np.zeros(a.shape[0])
c = np.hstack([a, b])