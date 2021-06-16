import os
import cv2
import pandas as pd
from numpy import asarray
from sklearn.decomposition import PCA as RandomizedPCA
from skimage import io
import imagecodecs


#Reading images from folder as NumpyNDArray
def load_images_from_folder(folder):
    image_list = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder, filename))
        if img is not None:
            array = asarray(img)
            image_list.append(array)
    #image_list_dataframe = pd.DataFrame(image_list)
    return image_list

listimg = load_images_from_folder("../Data/N2DH-GOWT1/img")
listgt = load_images_from_folder("../Data/N2DH-GOWT1/gt/tif")

#apply pca to images
def convert_pca(image_list):
    pca_list = []
    for image in listimg:
        pca = RandomizedPCA(300).fit(image.data)
        components = pca.transform(image.data)
        projected = pca.inverse_transform(components)
        if projected is not None:
            pca_list.append(projected)
    return pca_list

pca_listimg_trial = convert_pca(listimg)
pca_listgt_trial = convert_pca(listgt)

#pca image has 1024x1024 pixels -> gt can be used originally
    #test = pca_listimg_trial[1]
    #print(test)
    #print(test.shape)

# test if all images are included and went through pca and show first and last
    #print(pca_listimg_trial)
cv2.imshow('img1', listimg[0])
    #cv2.imshow('img6', listimg[5])
    #cv2.imshow('Trial1', pca_listimg_trial[0])
    #cv2.imshow('Trial6', pca_listimg_trial[5])
cv2.waitKey()


#cv2.imshow("Bild", pca_listimg_trial)
#cv2.waitKey()

def dataframe(pca_listimg):
    df = pd.DataFrame(pca_listimg)
    return df