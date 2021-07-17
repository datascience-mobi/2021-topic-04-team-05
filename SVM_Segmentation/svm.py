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
from SVM_Segmentation import tiles
from SVM_Segmentation import dicescore as ds
from sklearn.metrics import f1_score as dice_score
from SVM_Segmentation import array_to_img as ai
#workdir = os.path.normpath("/Users/laurasanchis/PycharmProjects/2021-topic-04-team-05/")
IMGSIZE = 50

def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0] #number of pixels = number of data points
    distances = 1 - Y * (np.dot(X, W)) #dot product=classification of pixel, multiplied with Y can be positiv (when
    # th prediction and gt is equal) or negative ()
    distances = np.maximum(0, distances)  # equivalent to max(0, distance)
    hinge_loss = regularization_strength * (np.sum(distances) / N) #durch N -> normalisierung, * reg strength,
    # wie wichtig die richtige klassifizierung ist
    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss #W^2 damit es nicht zu groß wird, hinge loss summiert damit die cost
    # größer wird (und penalization)
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])  # gives multidimensional array, 1 pixel con todas las features que tiene como
        # columnas

    distance = 1 - (Y_batch * np.dot(X_batch, W)) #clasifica el pixel
    dw = np.zeros(len(W)) #crea vector de weights vacío para ir llenándolo

    for ind, d in enumerate(distance):
        if max(0, d) == 0: #si d es negativo -> pixel clasificado bien, se mantiene el weight calculado
            di = W
        else:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind]) #cuando pixel clasificado mal,
            # multiplicas c por la derivada de distance (y*np.dot(x,w)), para ver cuánto quieres ir en la dirección
            # de la derivada para corregir la clasificación
        dw += di #derivada w

    dw = dw/len(Y_batch)  # average
    return dw

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
    img = resize(img, (imgSize, imgSize))
    #img = tiles.tiles(img, imgSize)
    #img = ai.oneD_array_to_twoD_array(img)
    img_canny = canny(img)
    img_canny = img_canny.reshape(-1, 1)
    img = img.reshape(-1, 1)
    bias_term = np.ones(img.shape[0]).reshape(-1, 1)
    return np.hstack([img, img_canny, bias_term])

def processMask(image_path, imgSize):
    img = io.imread(image_path)
    #img = tiles.tiles(img, imgSize)
    #img = ai.oneD_array_to_twoD_array(img)
    img = resize(img, (imgSize, imgSize))
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
    return dice_score(pred, gt)

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
X_test = np.vstack([processImage(imgPath, IMGSIZE) for imgPath in imgs[NImagesTraining:]])
y_test = np.concatenate([processMask(imgPath, IMGSIZE) for imgPath in masks[NImagesTraining:]])

skf = StratifiedKFold(n_splits=2)

regularization_strength = 10000
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
plt.savefig(f"{output_dir}/lr-{learning_rate}-reg-{regularization_strength}.png")

img_names = []
for filename in sorted(os.listdir('../Data/N2DH-GOWT1/img')):
        img_names.append(filename)

#Ntest = len(imgs) - NImagesTraining
fig, ax = plt.subplots(dpi=90)
for i in range(np.int(X_test.shape[0]/IMGSIZE)):
    if i == 1:
        img = X_test[i:IMGSIZE*i+i]
        pred, gt = predict(i, w_mean_model)
        ax.imshow(pred2Image(pred), cmap='gray')
        ax.axis('On')
        ax.set_title(f"Test img: {i + 1} Dice:{round(dice_score(gt, pred), 2)}")
        plt.savefig(f"{output_dir}/{img_names[i]}_pred_lr-{learning_rate}-reg-{regularization_strength}.png")
    if i > 1:
        img = X_test[IMGSIZE*(i-1)+i:IMGSIZE*i+i]
        pred, gt = predict(i, w_mean_model)
        ax.imshow(pred2Image(pred), cmap='gray')
        ax.axis('On')
        ax.set_title(f"Test img: {i+1} Dice:{round(dice_score(gt, pred), 2)}")
        plt.savefig(f"{output_dir}/{img_names[i]}_pred_lr-{learning_rate}-reg-{regularization_strength}.png")