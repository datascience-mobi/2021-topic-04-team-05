import statistics
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy

output_dir = f'../../Data/boxblots'
ticks = [1, 2, 3, 4, 5, 6]
labels = ['None', 'All', 'Gauss', 'Otsu', 'Watershed', 'PCA']

# N2DH-GOWT1, LEARNING RATE 1E-07
fig = numpy.array(
    [[0.71, 0.72, 0.71, 0.38, 0.4], [0.67, 0.69, 0.68, 0.79, 0.71], [0.65, 0.76, 0.75, 0.47, 0.5], [0.67, 0.68, 0.68,
                                                                                                   0.79, 0.54], [0.74, 0.76,
                                                                                                     0.74, 0.51, 0.71],
     [0.69, 0.69, 0.68, 0.52, 0.53]])
fig = plt.boxplot(fig.transpose())  # transpose nur hier notwendig, damit jede column 1 plot ist
_ = plt.xticks(ticks, labels)
plt.savefig(f"{output_dir}/boxblot_N2DH-GOW1-lr-1e-07.png")
