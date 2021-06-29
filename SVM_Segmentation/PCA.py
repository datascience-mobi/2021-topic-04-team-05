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


if __name__ == '__main__':

    imageread1 = rm.read_image('../Data/N2DH-GOWT1/img')
    imagenames1 = rm.read_imagename('../Data/N2DH-GOWT1/img')
    pca1 = convert_pca(imageread1, 0.75)

    plt.imshow(pca1[0])
    plt.show()

    flattened = rm.image_flatten(pca1)
    data1 = rm.dataframe(flattened, imagenames1)
    #print(data1)