import statistics
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy


# calculate mean & sd of dice scores for one setting/algorithm workflow

def dice_stats(dice_scores):
    d_mean = sum(dice_scores) / len(dice_scores)
    d_sd = statistics.stdev(dice_scores)
    print('The mean of the dice scores for setting 1 is ' + str(d_mean))
    print('The standard deviation of the dice scores for setting 1 is ' + str(d_sd))


# boxplot dice scores for one setting/algorithm workflow

# example for one array (= one setting)
# a = numpy.array([[0.7, 0.66, 0.73, 0.86, 0.98, 0.65]])
# a_mean = sum(a) / len(a)
# a_sd = statistics.stdev(a)
# plt.boxplot(a)
# plt.xticks([1], ['Dice score for setting No. 1'])
# plt.show()


# N2DH-GOWT1, LEARNING RATE 1E-07
a = numpy.array(
    [[0.79, 0.71], [0.43, 0.46], [0.79, 0.71],
     [0.45, 0.48], [0.79, 0.71], [0.79, 0.71]])
plt.boxplot(a.transpose())  # transpose nur hier notwendig, damit jede column 1 plot ist
ticks = [1, 2, 3, 4, 5]
labels = ['All', 'Gauss', 'Otsu', 'Watershed', 'Synthetic Images', 'None', 'Tiles', 'PCA']
plt.xticks(ticks, labels)
plt.show()

# N2DH-GOWT1, LEARNING RATE 1E-05
b = numpy.array(
    [[0.75, 0.68], [0.36, 0.39]])
plt.boxplot(b.transpose())  # transpose nur hier notwendig, damit jede column 1 plot ist
ticks = [1, 2]
labels = ['All', 'None']
plt.xticks(ticks, labels)
plt.show()