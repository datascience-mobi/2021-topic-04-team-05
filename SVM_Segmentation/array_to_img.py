import readimages as rm
import numpy as np
import math

def oneD_array_to_twoD_array(ExtendedDataFrame):
    for i in range(len(ExtendedDataFrame)):
        twoDarray = np.stack(ExtendedDataFrame.iloc[i], axis=0)
        a = int(math.sqrt(len(twoDarray)))
        twoDarray = twoDarray.reshape(a, a)
    return twoDarray


if __name__ == '__main__':
    imageread1 = rm.read_image('../Data/N2DH-GOWT1/img')
    imagenames1 = rm.read_imagename('../Data/N2DH-GOWT1/img')
    imageflattened1 = rm.image_flatten(imageread1)
    data1 = rm.dataframe(imageflattened1, imagenames1)
    ExtendedDataFrame = data1[np.repeat(data1.columns.values, 4)]  # 4 is the numbers of array size after PCA and tiles
    print(oneD_array_to_twoD_array(ExtendedDataFrame))
