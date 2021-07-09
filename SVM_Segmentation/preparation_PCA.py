import pandas as pd
import sklearn.decomposition as skdecomp
from sklearn.preprocessing import StandardScaler
import readimages as rm
from matplotlib import pyplot as plt

def convert_pca(image_dataframe, variance):
    """
    This function standardizes and scales the 2D-numpy-ndarray, so that it can directly be used for finding its
    principal components. It then transforms it back to its original shape.
    :param image_dataframe:
    :PCs: number of principal components
    :return:
    """
    pca_list = []
    for image in image_dataframe:
        image = StandardScaler().fit_transform(image)
        pca = skdecomp.PCA(variance)
        pca.fit(image)
        components = pca.transform(image)
        projected = pca.inverse_transform(components)
        if projected is not None:
            pca_list.append(projected)
    return pca_list

def pca(image_list, principal_components, variance):

    pca_list = []
    for image in image_list:
        image = image.reshape(1, -1)
        image_df = pd.DataFrame(image)
        new_image_pca_dataframe = pd.concat([image_df] * principal_components)
            #pca_data = rm.dataframe(pca_list)
        new_image_pca_dataframe = new_image_pca_dataframe.values
        pca = skdecomp.PCA()
        pca.fit(new_image_pca_dataframe)
        components = pca.transform(new_image_pca_dataframe)
        pca_image = components[0, :]
        pca_list.append(pca_image)
    return pca_list



if __name__ == '__main__':

    imageread1 = rm.read_image('../Data/N2DH-GOWT1/img')
    imagenames1 = rm.read_imagename('../Data/N2DH-GOWT1/img')
    pca1 = convert_pca(imageread1, 0.75)

    #plt.imshow(pca1[0])
    #plt.show()

    #flattened = rm.image_flatten(pca1)
    #data1 = rm.dataframe(flattened, imagenames1)
    #print(data1)

    flattened_image = rm.image_flatten(imageread1)
    dataframe = rm.dataframe(flattened_image, imagenames1)
    pca_list_images = pca(imageread1, 100, 0.9)
    print(pca_list_images)