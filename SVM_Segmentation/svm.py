import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import skimage.feature
from skimage import io
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
from skimage.feature import multiscale_basic_features, canny
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score as dice_score
from SVM_Segmentation.preprocessing import tiles
from SVM_Segmentation.evaluation import dicescore as ds
from SVM_Segmentation import pixel_conversion as pc

# workdir = os.path.normpath("/Users/laurasanchis/PycharmProjects/2021-topic-04-team-05/")
plt.rcParams["figure.figsize"] = (10, 5)

def compute_cost(weights, features, labels):
    """"
    This function calculates the cost for a given feature, label and the weights vector.
    First, the hinge loss is calculated, a loss function that determines the importance of
    misclassified pixels. The smaller the soft margin factor (smf) is, the smaller the hinge loss is, which means the
    SVM margin will be softer.
    The cost function depends on the weights vector and the hinge loss.
    :param weights: Weights vector
    :param features: Feature vector, has different columns for the different features for each pixel. Every pixel is
    a row.
    :param labels: Labels vector, is either 1 or -1 for each pixel (each row).
    :return: The cost of the current weights vector at classifying the pixels.
    """
    # calculate hinge loss
    number_pixels = features.shape[0]
    distances_to_hyperplane = 1 - labels * (np.dot(features, weights))
    distances_to_hyperplane = np.maximum(0, distances_to_hyperplane)
    hinge_loss = soft_margin_factor * (np.sum(distances_to_hyperplane) / number_pixels)

    # calculate cost
    cost = 1 / 2 * np.dot(weights, weights) + hinge_loss
    return cost


def calculate_cost_gradient(weights, features_pixel, label_pixel):
    """
    This funciton calculates the gradient of the cost function, so it makes its derivative, in order to know in which
    direction the gradient descent has to go to find the minimum of the cost function.
    :param weights: weights vector.
    :param features_pixel: because we chose SGD, only one pixel will be passed to this function. This means,
    only one row with different columns depending on the amount of features.
    :param label_pixel: label of the current pixel, either +1 or -1.
    :return: gradient of the cost function with that weight vector.
    """
    # In our case, we will use Stochastic Gradient Descent, so only one pixel will be passed.
    # Because of this, its label is only one number.
    if type(label_pixel) == np.float64:
        label_pixel = np.array([label_pixel])
        features_pixel = np.array([features_pixel])

    # Calculate distance to hyperplane, to classify the pixel.
    distance_to_hyperplane = 1 - (label_pixel * np.dot(features_pixel, weights))

    # Create an empty gradient vector, to fill with the gradient of the current pixel.
    gradient_cost = np.zeros(len(weights))

    for index_pixel, distance_pixel in enumerate(distance_to_hyperplane):
        # For correctly classified pixels, the current weight vector is maintained
        if max(0, distance_pixel) == 0:
            gradient_pixel = weights
        # For incorrectly classified pixels, the weight vector is corrected.
        else:
            gradient_pixel = weights - (soft_margin_factor * label_pixel[index_pixel] * features_pixel[index_pixel])
        gradient_cost += gradient_pixel

    return gradient_cost


def sgd(features, labels):
    """
    This function calculates the stochastic gradient descent to minimize our cost function.
    :param features: all pixels as rows, with n columns, which stand for n features.
    :param labels: all pixels as rows, only one column per pixel, either +1 or -1.
    :return: the weight vector found at the minimum of the cost function, and the history of how the costs have
    evolved during the calculations.
    """
    # Maximum number of cycles to try to find the minimum of the cost function.
    max_epochs = 100
    # Create an empty weight vector that is the same size as the number of columns (features) of a single pixel.
    weights = np.zeros(features.shape[1])
    # Define the first cost for our function and an empty history cost list.
    prev_cost = float("inf")
    history_cost = []
    patience = 0

    # Stochastic gradient descent
    for epoch in range(0, max_epochs):
        # shuffle to prevent repeating update cycles
        features_shuffled, labels_shuffled = shuffle(features, labels)
        for pixel_index, pixel_value in enumerate(features_shuffled):
            gradient = calculate_cost_gradient(weights, pixel_value, labels_shuffled[pixel_index])
            weights = weights - (learning_rate * gradient)

        # Calculate cost to evaluate the advance.
        cost = compute_cost(weights, features, labels)
        history_cost.append(cost)
        if epoch % 20 == 0 or epoch == max_epochs - 1:
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))

        # Stoppage criterion
        if prev_cost < cost:
            if patience == 10:
                return weights, history_cost
            else:
                patience += 1
        else:
            patience = 0
        prev_cost = cost

    return weights, history_cost


def process_image(image_path, img_size):
    img = io.imread(image_path)
    # img = resize(img, (img_size, img_size))
    img = tiles.tiles(img, img_size)
    img = pc.one_d_array_to_two_d_array(img)
    img_canny = canny(img)
    img_canny = img_canny.reshape(-1, 1)
    img = img.reshape(-1, 1)
    bias_term = np.ones(img.shape[0]).reshape(-1, 1)
    return np.hstack([img, img_canny, bias_term])


def process_mask(image_path, img_size):
    img = io.imread(image_path)
    img = tiles.tiles(img, img_size)
    img = pc.one_d_array_to_two_d_array(img)
    # img = resize(img, (img_size, img_size))
    img[img > 0] = 1
    img[img < 1] = -1
    img = img.flatten()
    return img


def predict(image_index, weights):
    processed_img = process_image(imgs[image_index], size)
    prediction = [np.sign(np.dot(processed_img[pixelN], weights)) for pixelN in range(processed_img.shape[0])]
    ground_truth = process_mask(masks[image_index], size)
    return prediction, ground_truth


def predict_score(img, gt, weights):
    pred = [np.sign(np.dot(img[pixelN], weights)) for pixelN in range(img.shape[0])]
    return dice_score(pred, gt)


def pred2image(prediction):
    prediction = np.array(prediction)
    predsize = int(np.sqrt(len(prediction)))
    return prediction.reshape((predsize, predsize))


    soft_margin_factor = 10000
    learning_rate = 0.00001
    size = 50

def svm(dataset, n_train, soft_margin_factor, learning_rate, splits, size, gauss: bool=True,
        watershed: bool=True):
    imgs = sorted(glob(f"../Data/{dataset}/img/*.tif"))
    masks = sorted(glob(f"../Data/{dataset}/gt/tif/*.tif"))
    print(f"{len(imgs)} images detected and {len(masks)} masks detected")

    NImagesTraining = n_train
    X_train = np.vstack([process_image(imgPath, size) for imgPath in imgs[:NImagesTraining]])
    y_train = np.concatenate([process_mask(imgPath, size) for imgPath in masks[:NImagesTraining]])
    X_test = [process_image(imgPath, size) for imgPath in imgs[NImagesTraining:]]
    y_test = [process_mask(imgPath, size) for imgPath in masks[NImagesTraining:]]

    skf = StratifiedKFold(n_splits=splits)

    model = {}
    for split_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        X_train_split, X_test_split = X_train[train_index], X_train[test_index]
        y_train_split, y_test_split = y_train[train_index], y_train[test_index]
        w, hist = sgd(X_train_split, y_train_split)
        dice = predict_score(X_test_split, y_test_split, w)
        model[split_number] = {"train_index": train_index, "test_index": test_index, "w": w, "hist": hist, "dice": dice,
                              "image_size": size}
        print(model[split_number]["w"])
        print(model[split_number]["dice"])

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
        ii = i + NImagesTraining
        pred, gt = predict(ii, w_mean_model)
        ax.imshow(pred2image(pred), cmap='gray')
        ax.axis('On')
        ax.set_title(f"Test img: {ii + 1} Dice:{round(dice_score(gt, pred), 2)}")
        plt.savefig(f"{output_dir}/{img_names[ii]}_pred_lr-{learning_rate}-reg-{soft_margin_factor}.png")
