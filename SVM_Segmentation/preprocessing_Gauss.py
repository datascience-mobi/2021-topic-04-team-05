import matplotlib.pyplot as plt
import cv2
import os

def gauss_filter(path_to_folder):
    """
      This function filters images with a lot of noises
      :param path to folder:
      :return:
      """
    images = []
    for filename in os.listdir(path_to_folder): #read each image from a folder
        original_image = cv2.imread(os.path.join(path_to_folder,filename)) #apply the gauss filter
        filtered = cv2.GaussianBlur(original_image, (5, 5), 0)
        if original_image is not None:
            images.append(filtered)
    return images

if __name__ == '__main__':

    path = ("/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DL-HeLa/gt")
    max = os.listdir(path)
    for i in range (1, len(max)):
        segmented_images = gauss_filter(path)
        plt.imshow(segmented_images[i])
        plt.show()







