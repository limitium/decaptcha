import glob
import numpy as np
from scipy import ndimage
from itertools import groupby
import matplotlib.pyplot as plt

THRESHOLD_COLOR_LINE = 150

files = glob.glob("./straight_small/*.jpg")

hist = np.zeros((len(files), 11), dtype=float)
f = 0
for name in files:
    img = ndimage.imread(name, flatten=False, mode='L')
    binary_img = img < THRESHOLD_COLOR_LINE
    mask = binary_img.astype(int)
    for y_line in mask.T:
        if 1 in y_line:
            max_line = max(sum(g) for v, g in groupby(y_line) if v)
            hist[f][max_line] += 1
    f += 1

median = np.max(hist.T, axis=1)
print median
median = np.median(hist.T, axis=1)
print median
plt.hist(median, 26)
plt.show()
