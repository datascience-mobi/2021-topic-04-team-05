import os
import cv2
from numpy import asarray
from sklearn.decomposition import PCA as RandomizedPCA
from PIL import Image

#Reading images from folder as NumpyNDArray
def load_images_from_folder(folder):
    image_list = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            array = asarray(img)
            image_list.append(array)
    return image_list

listimg = load_images_from_folder("/Users/juanandre/PycharmProjects/svm/Data/N2DH-GOWT1/img")
listgt = load_images_from_folder("/Users/juanandre/PycharmProjects/svm/Data/N2DH-GOWT1/gt/tif")


#apply pca to images
def convert_pca(image_list):
    for image in image_list:
        pca = RandomizedPCA(1000).fit(image.data)
        components = pca.transform(image.data)
        projected = pca.inverse_transform(components)
        if projected is not None:
            array = asarray(projected)
            list_pca = image_list.append(array)
    return list_pca

pca_listimg_trial = convert_pca(listimg)
pca_listgt_trial = convert_pca(listgt)

#print(pca_listimg_trial[1].shape)

#cv2.imshow("Bild", pca_listimg_trial)
#cv2.waitKey()

def dataframe(pca_listimg):
    df = pd.DataFrame(pca_listimg)
    return df
