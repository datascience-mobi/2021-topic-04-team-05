import matplotlib.pyplot as plt
import numpy

def dice_stats(dice_scores):
    d_mean = sum(dice_scores) / len(dice_scores)
    d_sd = statistics.stdev(dice_scores)
    print('The mean of the dice scores for setting 1 is ' + str(d_mean))
    print('The standard deviation of the dice scores for setting 1 is ' + str(d_sd))

# N2DH-GOWT1, LEARNING RATE 1E-07
fig = numpy.array(
    [[0.71, 0.72, 0.71, 0.38, 0.4], [0.67, 0.69, 0.68, 0.79, 0.71], [0.65, 0.76, 0.75, 0.47, 0.5],
     [0.67, 0.68, 0.68, 0.79, 0.54], [0.74, 0.76, 0.74, 0.51, 0.71], [0.69, 0.69, 0.68, 0.52, 0.53]])
fig = plt.boxplot(fig.transpose())
_ = plt.ylabel("Dice score")
_ = plt.xlabel("Features")
__ = plt.xticks(ticks, labels)

plt.savefig(f"{output_dir}/boxplot_N2DH-GOW1-lr-1e-07.png")

# NIH3T3, LEARNING RATE 1E-07
Otsu = [0.85, 0.81, 0.71, 0.74, 0.66, 0.66, 0.74, 0.05, 0.46, 0.0, 0.7, 0.06, 0.6, 0.64, 0.13, 0.1, 0.78]
print(dice_stats(Otsu))
output_dir = f'../../Data/boxplots'
ticks = [1, 2, 3, 4, 5, 6]
labels = ['None', 'All', 'Gauss', 'Otsu', 'Watershed', 'PCA']

fig = numpy.array(
    [[0.76, 0.43, 0.5, 0.76, 0.64, 0.46, 0.52, 0.53, 0.41, 0.57, 0.5, 0.36, 0.59, 0.62, 0.41, 0.33, 0.54],
     [0.85, 0.8, 0.74, 0.75, 0.68, 0.58, 0.76, 0.41, 0.54, 0.49, 0.7, 0.5, 0.63, 0.64, 0.63, 0.63, 0.79],
     [0.8, 0.43, 0.47, 0.77, 0.61, 0.46, 0.51, 0.48, 0.41, 0.51, 0.46, 0.35, 0.58, 0.59, 0.39, 0.33, 0.53],
     [0.85, 0.81, 0.71, 0.74, 0.66, 0.66, 0.74, 0.05, 0.46, 0.0, 0.7, 0.06, 0.6, 0.64, 0.13, 0.1, 0.78],
     [0.82, 0.43, 0.47, 0.76, 0.62, 0.46, 0.51, 0.5, 0.41, 0.53, 0.47, 0.36, 0.58, 0.61, 0.39, 0.33, 0.54],
     [0.86, 0.58, 0.73, 0.75, 0.71, 0.54, 0.66, 0.6, 0.49, 0.7, 0.67, 0.43, 0.67, 0.67, 0.6, 0.47, 0.67]])
fig = plt.boxplot(fig.transpose()) 
_ = plt.ylabel("Dice score")
_ = plt.xlabel("Features")
__ = plt.xticks(ticks, labels)
plt.savefig(f"{output_dir}/boxplot_NIH3T3-lr-1e-07.png")


# N2DL-HeLa, LEARNING RATE 1E-07
a = numpy.array(
    [[0, 0, 0], [0.59, 0.64, 0.64], [0.7, 0.43, 0.44],
     [0.56, 0.65, 0.65], [0, 0, 0], [0.7, 0.44, 0.44]])

output_dir = f'../../Data/boxblots'
plt.boxplot(a.transpose())  # transpose nur hier notwendig, damit jede column 1 plot ist
ticks = [1, 2, 3, 4, 5, 6]
labels = ['None', 'All', 'Gauss', 'Otsu', 'Watershed', 'PCA']
plt.ylabel("Dice score")
plt.xlabel("Features")
plt.xticks(ticks, labels)
plt.show()
plt.savefig(f"{output_dir}/boxblot_N2DL-HeLa-lr-1e-07.png")
