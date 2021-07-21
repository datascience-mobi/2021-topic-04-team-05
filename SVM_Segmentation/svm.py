import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score as dice_score
from SVM_Segmentation.preprocessing import watershed as ws
from SVM_Segmentation.preprocessing import cooperation_team03_otsu as ot
from SVM_Segmentation.preprocessing import pca

plt.rcParams["figure.figsize"] = (10, 5)

def compute_cost(weights, features, labels, soft_margin_factor):
    """"
    This function calculates the cost for a given feature, label and the weights vector.
    First, the hinge loss is calculated, a loss function that determines the importance of
    misclassified pixels. The smaller the soft margin factor (smf) is, the smaller the hinge loss is, which means the
    SVM margin will be softer.
    The cost function depends on the weights vector and the hinge loss.
    :param soft_margin_factor:
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


def calculate_cost_gradient(weights, features, labels, soft_margin_factor):
    """
    This function calculates the gradient of the cost function. It computes its derivative, in order to know in which
    direction the gradient descent has to go to find the minimum of the cost function.
    :param soft_margin_factor: determines the importance of misclassified pixels.
    :param weights: weights vector.
    :param features: a vector with all our pixels as rows.
    :param labels: label of the pixels, either +1 or -1.
    :return: gradient of the cost function with that weight vector.
    """
    # In order to iterate correctly, we turn our vectors into arrays.
    labels = np.array([labels])
    features = np.array([features])

    # Calculate distance to hyperplane, to classify the pixels. This dot product can be changed depending on the
    # chosen kernel.
    distance_to_hyperplane = 1 - (labels * np.dot(features, weights))

    # Create an empty gradient vector, to fill with the gradient of the current pixel.
    gradient_cost = np.zeros(len(weights))

    for index_pixel, distance_pixel in enumerate(distance_to_hyperplane):
        # For correctly classified pixels, the current weight vector is maintained
        if max(0, distance_pixel) == 0:
            gradient_pixel = weights
        # For incorrectly classified pixels, the weight vector is corrected in the direction contrary to the gradient.
        else:
            gradient_pixel = weights - (soft_margin_factor * labels[index_pixel] * features[index_pixel])
        gradient_cost += gradient_pixel

    return gradient_cost


def sgd(features, labels, soft_margin_factor, learning_rate, max_epochs):
    """
    This function calculates the stochastic gradient descent to minimize our cost function.
    :param features: all pixels as rows, with n columns, which stand for n features.
    :param labels: all pixels as rows, only one column per pixel, either +1 or -1.
    :param soft_margin_factor:
    :param learning_rate:
    :param max_epochs: maximum number of cycles to try to find the minimum of the cost function.
    :return: the weight vector found at the minimum of the cost function, and the history of how the costs have
    evolved during the calculations.
    """
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
            gradient = calculate_cost_gradient(weights, pixel_value, labels_shuffled[pixel_index], soft_margin_factor)
            weights = weights - (learning_rate * gradient)

        # Calculate cost to evaluate the advance.
        cost = compute_cost(weights, features, labels, soft_margin_factor)
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


def process_image(image_path, img_size, Otsu: bool = False, Watershed: bool = False, Gauss: bool = False,
                   PCA: bool = False):
    """
    Processes our image so that the dimensions are all the same, its intensities are normalized and different
    features are added.
    :param image_path: path to image.
    :param img_size: parameter for resizing.
    :param Otsu: activation of otsu filter.
    :param Watershed: activates watershed segmentation.
    :param Gauss: activates gauss filter
    :param PCA: activates PCA.
    :return: image with all normalized pixels as one column.
    """
    img = io.imread(image_path)

    filter_list = []
    if Otsu == True:
        img_otsu = ot.complete_segmentation(img, intensity_lvls=2 ** 16)
        img_otsu = resize(img_otsu, (img_size, img_size))
        img_otsu = img_otsu.reshape(-1, 1)
        filter_list.append(img_otsu)

    if Watershed == True:
        img_watershed = ws.watershed(image_path)
        img_watershed = resize(img_watershed, (img_size, img_size))
        img_watershed = img_watershed.reshape(-1, 1)
        filter_list.append(img_watershed)

    if Gauss == True:
        img_gauss = gaussian(img, sigma=2)
        img_gauss = resize(img_gauss, (img_size, img_size))
        img_gauss = img_gauss.reshape(-1, 1)
        filter_list.append(img_gauss)

    if PCA == True:
        img_pca = pca.convert_pca(img, 0.9)
        img_pca = resize(img_pca, (img_size, img_size))
        img_pca = img_pca.reshape(-1, 1)
        filter_list.append(img_pca)

    img = resize(img, (img_size, img_size))
    img = img.reshape(-1, 1)
    bias_term = np.ones(img.shape[0]).reshape(-1, 1)
    if len(filter_list) == 0:
        stacked_dataframe = np.hstack([img, bias_term])
    if len(filter_list) == 1:
        stacked_dataframe = np.hstack([img, filter_list[0], bias_term])
    if len(filter_list) == 2:
        stacked_dataframe = np.hstack([img, filter_list[0], filter_list[1], bias_term])
    if len(filter_list) == 3:
        stacked_dataframe = np.hstack([img, filter_list[0], filter_list[1], filter_list[2], bias_term])
    if len(filter_list) == 4:
        stacked_dataframe = np.hstack([img, filter_list[0], filter_list[1], filter_list[2], filter_list[3], bias_term])
    return stacked_dataframe


def process_mask(image_path, img_size):
    """
    Processes the mask so that it has the same dimensions as the original image and is normalized.
    :param image_path: mask path.
    :param img_size: resizing.
    :return: normalized and resized mask.
    """
    img = io.imread(image_path)
    img = resize(img, (img_size, img_size))
    img[img > 0] = 1
    img[img < 1] = -1
    img = img.flatten()
    return img


def predict(dataset, image_index, weights, size, datatype, Otsu: bool = False, Watershed: bool = False, Gauss: bool = False,
            PCA: bool = False):
    """
    Calculates the prediction of the image with the svm-determined weights vector.
    :param dataset: chooses dataset.
    :param image_index: chooses image within dataset.
    :param weights: weights vector.
    :param size: parameter for resizing.
    :return:
    """
    imgs = sorted(glob(f"../Data/{dataset}/img/*.{datatype}"))
    masks = sorted(glob(f"../Data/{dataset}/gt/{datatype}/*.{datatype}"))
    processed_img = process_image(imgs[image_index], size, Otsu, Watershed, Gauss, PCA)
    prediction = [np.sign(np.dot(processed_img[pixelN], weights)) for pixelN in range(processed_img.shape[0])]
    ground_truth = process_mask(masks[image_index], size)
    return prediction, ground_truth


def predict_score(img, gt, weights):
    """
   Calculates dice score of our prediciton vs. the ground truth.
   :param img: image that we want to segment.
   :param gt: ground truth corresponding to segmented image.
   :param weights: weights vector that should classify pixels.
   :return: dice score from the prediction compared to the ground truth.
   """
    pred = [np.sign(np.dot(img[pixelN], weights)) for pixelN in range(img.shape[0])]
    return dice_score(pred, gt)


def pred2image(prediction):
    """
    Creates an image from prediction array.
    :param prediction: uses the prediction array created from our segmentation.
    :return: segmented image.
    """
    prediction = np.array(prediction)
    predsize = int(np.sqrt(len(prediction)))
    return prediction.reshape((predsize, predsize))


def svm(dataset, n_train, soft_margin_factor, learning_rate, splits, size, max_epochs, filters_name, datatype, Otsu: bool = False,
        Watershed: bool = False, Gauss: bool = False, PCA: bool = False):
    """
    Trains and tests our support vector machine using a specific dataset.
    :param dataset: name of the dataset, can be N2DH-GOWT1, N2DL-HeLa or NIH3T3.
    :param n_train: Amount of pictures in that dataset that are used for training (usually 2/3 of the images).
    :param soft_margin_factor: How strong or soft our margin is.
    :param learning_rate: how big the steps are in the direction contrary to the gradient.
    :param splits: Amount of splits in the cross validation.
    :param size: Resizing of the images, what amount of pixels they should have.
    :param max_epochs: maximum amount of epochs for our stochastic gradient descent.
    :param filters_name: in order to save images with the correct filter name.
    :param Otsu: activates otsu filter.
    :param Watershed: activates watershed segmentation.
    :param Gauss: activates gauss filter
    :param PCA: activates PCA.
    :return: segmented images are saved into "pred" folder in the corresponding dataset.
    """
    # Reading in our images
    imgs = sorted(glob(f"../Data/{dataset}/img/*.{datatype}"))
    masks = sorted(glob(f"../Data/{dataset}/gt/tif/*.{datatype}"))
    print(f"{len(imgs)} images detected and {len(masks)} masks detected")

    # Defining our train set
    NImagesTraining = n_train
    X_train = np.vstack([process_image(imgPath, size, Otsu, Watershed, Gauss, PCA) for imgPath in imgs[:NImagesTraining]])
    y_train = np.concatenate([process_mask(imgPath, size) for imgPath in masks[:NImagesTraining]])

    # Cross validation
    skf = StratifiedKFold(n_splits=splits)
    model = {}
    for split_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        X_train_split, X_test_split = X_train[train_index], X_train[test_index]
        y_train_split, y_test_split = y_train[train_index], y_train[test_index]
        w, hist = sgd(X_train_split, y_train_split, soft_margin_factor, learning_rate, max_epochs)
        dice = predict_score(X_test_split, y_test_split, w)
        model[split_number] = {"train_index": train_index, "test_index": test_index, "w": w, "hist": hist, "dice": dice,
                               "image_size": size}
        print(model[split_number]["w"])
        print(model[split_number]["dice"])
    w_mean_model = np.mean([model[i]["w"] for i in model.keys()], axis=0)

    # Plotting the cost history
    output_dir = f'../Data/{dataset}/pred'
    for i in range(len(model.keys())):
        fig = plt.plot(model[i]['hist'], label=f"Split {i}")
        _ = plt.ylabel("Cost function")
        _ = plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{output_dir}/lr-{learning_rate}-reg-{soft_margin_factor}-{filters_name}.png")

    # Plotting the segmented images from the test set
    img_names = []
    for filename in sorted(os.listdir(f'../Data/{dataset}/img')):
        img_names.append(filename)
    Ntest = len(imgs) - NImagesTraining
    fig, ax = plt.subplots(dpi=90)
    for i in range(Ntest):
        ii = NImagesTraining + i
        pred, gt = predict(dataset, ii, w_mean_model, size, datatype, Otsu, Watershed, Gauss, PCA)
        ax.imshow(pred2image(pred), cmap='gray')
        ax.axis('On')
        ax.set_title(f"Test img: {ii + 1} Dice:{round(dice_score(gt, pred), 2)}")
        plt.savefig(f"{output_dir}/{img_names[ii]}_pred_lr-{learning_rate}-reg-"
                    f"{soft_margin_factor}-{filters_name}.png")



def synthetic_svm(dataset, synth_dataset, soft_margin_factor, learning_rate, splits, size, max_epochs, filters_name, datatype,
                  Otsu: bool = False, Watershed: bool = False, Gauss: bool = False, PCA: bool = False):
    """
    Segments images using synthetic images as training set and the rest of the images from the dataset that weren't
    used for the synthetic image generation as a test set.
    :param dataset: name of the dataset, can be N2DH-GOWT1, N2DL-HeLa or NIH3T3.
    :param synth_dataset: name of the synthetic dataset, can be N2DH-GOWT1_t01, N2DL-HeLa_t13 or NIH3T3_dna-0.
    :param soft_margin_factor: How strong or soft our margin is.
    :param learning_rate: how big the steps are in the direction contrary to the gradient.
    :param splits: Amount of splits in the cross validation.
    :param size: Resizing of the images, what amount of pixels they should have.
    :param max_epochs: maximum amount of epochs for our stochastic gradient descent.
    :param filters_name: in order to save images with the correct filter name.
    :param datatype: png or tif.
    :param Otsu: activates otsu filter.
    :param Watershed: activates watershed segmentation.
    :param Gauss: activates gauss filter
    :param PCA: activates PCA.
    :return: segmented images are saved into "pred" folder in the corresponding dataset.
    """
    # Reading in our images
    imgs = sorted(glob(f"../Data/{dataset}/img/*.{datatype}"))
    synth_imgs = sorted(glob(f"../Data/synthetic_cell_images/{synth_dataset}/generated_images_img/*.tif"))
    masks = sorted(glob(f"../Data/{dataset}/gt/{datatype}/*.{datatype}"))
    synth_masks = sorted(glob(f"../Data/synthetic_cell_images/{synth_dataset}/generated_images_gt/*.tif"))
    print(f"{len(synth_imgs)} synthetic images and {len(synth_masks)} synthetic masks detected for training")
    print(f"{len(imgs)} images and {len(masks)} masks detected for testing")

    # Defining our train set
    X_train = np.vstack([process_image(imgPath, size, Otsu, Watershed, Gauss, PCA) for imgPath in synth_imgs])
    y_train = np.concatenate([process_mask(imgPath, size) for imgPath in synth_masks])

    # Cross validation
    skf = StratifiedKFold(n_splits=splits)

    model = {}
    for split_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        X_train_split, X_test_split = X_train[train_index], X_train[test_index]
        y_train_split, y_test_split = y_train[train_index], y_train[test_index]
        w, hist = sgd(X_train_split, y_train_split, soft_margin_factor, learning_rate, max_epochs)
        dice = predict_score(X_test_split, y_test_split, w)
        model[split_number] = {"train_index": train_index, "test_index": test_index, "w": w, "hist": hist, "dice": dice,
                               "image_size": size}
        print(model[split_number]["w"])
        print(model[split_number]["dice"])

    w_mean_model = np.mean([model[i]["w"] for i in model.keys()], axis=0)

    # Plotting the cost history
    output_dir = f'../Data/{dataset}/pred'

    for i in range(len(model.keys())):
        fig = plt.plot(model[i]['hist'], label=f"Split {i}")
        _ = plt.ylabel("Cost function")
        _ = plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{output_dir}/lr-{learning_rate}-reg-{soft_margin_factor}-{filters_name}.png")

    # Plotting the segmented images from the test set
    img_names = []
    for filename in sorted(os.listdir(f'../Data/{dataset}/img')):
        img_names.append(filename)

    Ntest = len(imgs)
    fig, ax = plt.subplots(dpi=90)
    for i in range(Ntest):
        pred, gt = predict(dataset, i, w_mean_model, size, datatype, Otsu, Watershed, Gauss, PCA)
        ax.imshow(pred2image(pred), cmap='gray')
        ax.axis('On')
        ax.set_title(f"Test img: {i + 1} Dice:{round(dice_score(gt, pred), 2)}")
        plt.savefig(f"{output_dir}/{img_names[i]}_pred_lr-{learning_rate}-reg-"
                    f"{soft_margin_factor}-{filters_name}.png")


if __name__ == '__main__':
    # Segments the first dataset using a regularization strength of 10000, Learning rate of 1e07, 5 splits for cross
    # validation, 40 epochs as maximum and all filters.
    synthetic_svm("N2DH-GOWT1", "N2DH-GOWT1_t01", 10000, 0.0000001, 5, 250, 40, "All", "tif", Otsu=True,
                  Watershed=True, Gauss= True, PCA=True)



