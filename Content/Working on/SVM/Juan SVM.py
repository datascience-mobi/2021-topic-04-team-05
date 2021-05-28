#SVM Time!!

#Import Numpy for data manipulation
import numpy as np
#Matplotlib is for drawing graphs and colors
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#Load and manipulate data on One-Hot-Encoding
import pandas as pd
#Downsample the dataset
from sklearn.utils import resample
#Support Vector Machine for Classification
from sklearn.svm import SVC
from sklearn import svm
from sklearn import datasets
from sklearn.datasets import load_iris
#Cross Validation
from sklearn.model_selection import GridSearchCV
#Creates and draw confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
#To perform PCA to plot the data
from sklearn.decomposition import PCA

#Import the data to play with
#Iris is tool used in manipulating multidimensional earth science data

bc = datasets.load_breast_cancer()
df = pd.DataFrame(data=bc.data)
df["label"] = bc.target
##Scatter plot shown in fig 1
plt.scatter(df[0][df["label"] == 0], df[1][df["label"] == 0],
color='red', marker='o', label='malignant')
plt.scatter(df[0][df["label"] == 1], df[1][df["label"] == 1],
color='green', marker='*', label='benign')
plt.xlabel('Malignant')
plt.ylabel('Benign')
plt.legend(loc='upper left')
plt.show()

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(float)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

# fit the model
for kernel in ('linear', 'rbf', 'poly'):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure()
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()