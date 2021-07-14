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


# example for several arrays (= different setting)
d = numpy.array(
    [[0.7, 0.66, 0.73, 0.86, 0.98, 0.65], [0.7, 0.66, 0.73, 0.86, 0.98, 0.65], [0.7, 0.66, 0.73, 0.86, 0.98, 0.65],
     [0.7, 0.66, 0.73, 0.86, 0.98, 0.65]])
plt.boxplot(d.transpose())  # transpose nur hier notwendig, damit jede column 1 plot ist
ticks = [1, 2, 3, 4]
labels = ['Setting 1', 'Setting 2', 'Setting 3', 'Setting 4']
plt.xticks(ticks, labels)
# plt.show()
