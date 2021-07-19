# code for displaying multiple images in one figure

# import libraries
import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 2
columns = 3

# reading images
Image1 = cv2.imread('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img/t01.tif')
Image2 = cv2.imread('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img/t21.tif')
Image3 = cv2.imread('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img/t31.tif')
Image4 = cv2.imread('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img/t39.tif')
Image5 = cv2.imread('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img/t39.tif')
Image6 = cv2.imread('/Users/juanandre/PycharmProjects/2021-topic-04-team-05/Data/N2DH-GOWT1/img/t39.tif')

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("First")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(Image2)
plt.axis('off')
plt.title("Second")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(Image3)
plt.axis('off')
plt.title("Third")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(Image4)
plt.axis('off')
plt.title("Fourth")

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 5)

# showing image
plt.imshow(Image5)
plt.axis('off')
plt.title("Fifth")

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 6)

# showing image
plt.imshow(Image6)
plt.axis('off')
plt.title("Sixth")


plt.show()