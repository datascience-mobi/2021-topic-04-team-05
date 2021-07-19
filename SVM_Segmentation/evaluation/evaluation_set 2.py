# N2DL-HeLa, LEARNING RATE 1E-07
import numpy
from matplotlib import pyplot as plt

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