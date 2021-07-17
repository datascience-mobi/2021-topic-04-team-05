import numpy as np
from glob import glob
import skimage.feature
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
from skimage.feature import multiscale_basic_features, canny
from sklearn.model_selection import StratifiedKFold
from SVM_Segmentation.preprocessing import tiles
from SVM_Segmentation.evaluation import dicescore as ds
from sklearn.metrics import f1_score as dice_score
from SVM_Segmentation import pixel_conversion as ai
#workdir = os.path.normpath("/Users/laurasanchis/PycharmProjects/2021-topic-04-team-05/")
IMGSIZE = 50

def compute_cost(W, X, Y):
    """"
    This function calculates the cost for the given feature vector X, the label vector Y and the weights vector
    W. First, the hinge loss is calculated, which is a loss function that determines the importance/amount of
    misclassified pixels with the given weight vector. Depending on how big the soft margin factor (smf) is,
    the hinge loss is bigger or smaller for the same amount of misclassified pixels. The smaller the smf, the softer
    the SVM margin will be, as the hinge loss will be smaller and the cost will be less. The cost function calculated
    for the vectors is what will be optimized, and depends on the weights vector (we are looking for weights that are
    as small as possible) and the hinge loss (it should also be reduced)
    :param W: Weights vector
    :param X: Feature vector, has different columns for the different features for each pixel. Every pixel is another row.
    :param Y: Labels vector, is either 1 or -1 for each pixel (each row).
    :return:
    """
    # calculate hinge loss
    number_pixels = X.shape[0]
    distances_to_hyperplane = 1 - Y * (np.dot(X, W))
    distances_to_hyperplane = np.maximum(0, distances_to_hyperplane)
    hinge_loss = soft_margin_factor * (np.sum(distances_to_hyperplane) / number_pixels)
    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_pixel, Y_pixel):
    """

    :param W:
    :param X_pixel:
    :param Y_pixel:
    :return:
    """
    # In our case, we will use Stochastic Gradient Descent, so only one pixel will be passed.
    # Because of this, Y (its label) is only one number.
    if type(Y_pixel) == np.float64:
        Y_pixel = np.array([Y_pixel])
        X_pixel = np.array([X_pixel])

    # Calculate distance to hyperplane, to classify the pixel.
    distance_to_hyperplane = 1 - (Y_pixel * np.dot(X_pixel, W))

    # Create an empty weight vector, to fill with the corrected weights.
    derivative_w = np.zeros(len(W))

    for ind, d in enumerate(distance_to_hyperplane):
        # For correctly classified pixels, the current weight is maintained
        if max(0, d) == 0:
            di = W
        # For incorrectly classified pixels, the weight is corrected.
        else:
            di = W - (soft_margin_factor * Y_pixel[ind] * X_pixel[ind]) #cuando pixel clasificado mal,
            # multiplicas c por la derivada de distance (y*np.dot(x,w)), para ver cuánto quieres ir en la dirección
            # de la derivada para corregir la clasificación
        derivative_w += di

    return derivative_w

def sgd(features, outputs):
    max_epochs = 100
    weights = np.zeros(features.shape[1])
    prev_cost = float("inf")
    #cost_threshold = 0.01  # Lower -> Longer training and better results
    history_cost = []
    # stochastic gradient descent
    patience = 0
    for epoch in range(0, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent) #minimizar weights por el valor de learning rate en
            # dirección contraria al gradiente, para encontrar el minimo

        cost = compute_cost(weights, features, outputs) #calcula el coste para comparar la epoca anterior con la
        # actual y ver si esta avanzando
        history_cost.append(cost)
        if epoch % 20 == 0 or epoch == max_epochs - 1:
            #cost = compute_cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if (prev_cost < cost):
                #best_cost = min(history_cost)
                if patience == 10:
                    return weights, history_cost
                else:
                    patience += 1
            else:
                patience = 0
            prev_cost = cost

    return weights, history_cost


def processImage(image_path, imgSize):
    img = io.imread(image_path)
    #img = resize(img, (imgSize, imgSize))
    img = tiles.tiles(img, imgSize)
    img = ai.oneD_array_to_twoD_array(img)
    img_canny = canny(img)
    img_canny = img_canny.reshape(-1, 1)
    img = img.reshape(-1, 1)
    bias_term = np.ones(img.shape[0]).reshape(-1, 1)
    return np.hstack([img, img_canny, bias_term])

def processMask(image_path, imgSize):
    img = io.imread(image_path)
    img = tiles.tiles(img, imgSize)
    img = ai.oneD_array_to_twoD_array(img)
    #img = resize(img, (imgSize, imgSize))
    img[img > 0] = 1
    img[img < 1] = -1
    img = img.flatten()
    return img

def predict(imageIndex, W):
    data = processImage(imgs[imageIndex], IMGSIZE)
    prediction = [np.sign(np.dot(data[pixelN], W)) for pixelN in range(data.shape[0])]
    groundTruth = processMask(masks[imageIndex], IMGSIZE)
    return prediction, groundTruth

def predictScore(data, gt, W):
    pred = [np.sign(np.dot(data[pixelN], W)) for pixelN in range(data.shape[0])]
    return ds.dice_score(pred, gt)

def pred2Image(prediction):
    prediction = np.array(prediction)
    predsize = int(np.sqrt(len(prediction)))
    return prediction.reshape((predsize, predsize))


imgs = sorted(glob("../Data/N2DH-GOWT1/img/*.tif"))
masks = sorted(glob("../Data/N2DH-GOWT1/gt/tif/*.tif"))
print(f"{len(imgs)} images detected and {len(masks)} masks detected")

NImagesTraining = 4
X_train = np.vstack([processImage(imgPath, IMGSIZE) for imgPath in imgs[:NImagesTraining]])
y_train = np.concatenate([processMask(imgPath, IMGSIZE) for imgPath in masks[:NImagesTraining]])
X_test = [processImage(imgPath, IMGSIZE) for imgPath in imgs[NImagesTraining:]]
y_test = [processMask(imgPath, IMGSIZE) for imgPath in masks[:NImagesTraining]]

skf = StratifiedKFold(n_splits=5)

soft_margin_factor = 10000
learning_rate = 0.00001

model = {}
for splitnumber, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    X_train_split, X_test_split = X_train[train_index], X_train[test_index]
    y_train_split, y_test_split = y_train[train_index], y_train[test_index]
    w, hist = sgd(X_train_split, y_train_split)
    dice = predictScore(X_test_split, y_test_split, w)
    model[splitnumber] = {"train_index": train_index, "test_index": test_index, "w": w, "hist": hist, "dice": dice,
                          "image_size": IMGSIZE}
    print(model[splitnumber]["w"])
    print(model[splitnumber]["dice"])

dice_mean_model = np.mean([model[i]["dice"] for i in model.keys()])
w_mean_model = np.mean([model[i]["w"] for i in model.keys()], axis=0)

output_dir = '../Data/N2DH-GOWT1/pred'

for i in range(len(model.keys())):
    fig = plt.plot(model[i]['hist'], label=f"Split {i}")
    _ = plt.ylabel("Cost function")
    _ = plt.xlabel("Epoch")
plt.legend()
plt.savefig(f"{output_dir}/lr-{learning_rate}-reg-{soft_margin_factor}.png")

img_names = []
for filename in sorted(os.listdir('../Data/N2DH-GOWT1/img')):
        img_names.append(filename)

Ntest = len(imgs) - NImagesTraining
fig, ax = plt.subplots(dpi=90)
for i in range(Ntest):
    ii = i+NImagesTraining
    pred, gt = predict(ii, w_mean_model)
    ax.imshow(pred2Image(pred), cmap='gray')
    ax.axis('On')
    ax.set_title(f"Test img: {ii+1} Dice:{round(ds.dice_score(gt, pred), 2)}")
    plt.savefig(f"{output_dir}/{img_names[ii]}_pred_lr-{learning_rate}-reg-{soft_margin_factor}.png")