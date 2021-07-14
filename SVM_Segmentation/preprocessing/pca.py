import pandas as pd
import sklearn.decomposition as decomposition
from sklearn.preprocessing import StandardScaler
import readimages as rm
import matplotlib.pyplot as plt


def convert_pca(image_dataframe, variance):
    """
    This function standardizes and scales the 2D-numpy-array, so that it can directly be used for finding its
    principal components. It then transforms it back to its original shape.
    :param variance: selected variance
    :param image_dataframe: image in data frame
    :PCs: number of principal components
    :return:
    """
    pca_list = []
    for image in image_dataframe:
        image = StandardScaler().fit_transform(image)
        pca = decomposition.PCA(variance)
        pca.fit(image)
        components = pca.transform(image)
        projected = pca.inverse_transform(components)
        if projected is not None:
            pca_list.append(projected)
    return pca_list


def pca_self(image_list, principal_components, variance):
    pca_list = []
    for image in image_list:
        image = image.reshape(1, -1)
        image_df = pd.DataFrame(image)
        new_image_pca_dataframe = pd.concat([image_df] * principal_components)
        # pca_data = rm.dataframe(pca_list)
        new_image_pca_dataframe = new_image_pca_dataframe.values
        pca = decomposition.PCA()
        pca.fit(new_image_pca_dataframe)
        components = pca.transform(new_image_pca_dataframe)
        pca_image = components[0, :]
        pca_list.append(pca_image)
    return pca_list


def pca_different_images(image_path, variance):
    pca_list = []
    image_list = rm.read_image(image_path)
    image_names = rm.read_imagename(image_path)
    flattened = rm.image_flatten(image_list)
    df = rm.dataframe(flattened, image_names)
    arrays = df.values
    arrays_transposed = arrays.transpose()
    pca = decomposition.PCA(variance)
    pca.fit(arrays_transposed)
    components = pca.transform(arrays_transposed)
    return components


if __name__ == '__main__':
    image_read1 = rm.read_image('../Data/N2DH-GOWT1/img')
    image_names1 = rm.read_imagename('../Data/N2DH-GOWT1/img')
    pca1 = convert_pca(image_read1, 0.75)

    # plt.imshow(pca1[0])
    # plt.show()

    # flattened = rm.image_flatten(pca1)
    # data1 = rm.dataframe(flattened, image_names1)
    # print(data1)

    # flattened_image = rm.image_flatten(image_read1)
    # dataframe = rm.dataframe(flattened_image, image_names1)
    # pca_list_images = pca_self(image_read1, 100, 0.9)
    # print(pca_list_images)

    print(pca_different_images('../../Data/N2DH-GOWT1/img', 0.95).shape)