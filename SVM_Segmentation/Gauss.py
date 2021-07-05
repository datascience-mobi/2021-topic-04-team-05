import cv2
import readimages as rm


def gauss(path_of_image_folder):
 image_list = rm.read_image(path_of_image_folder)
 gauss_image_list = []
 for image in image_list:
     GaussianFiltered = cv2.GaussianBlur(image, (5,5),0)
     print(GaussianFiltered)
     return gauss_image_list

if __name__ == '__main__':
    gauss('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/gt/jpg')