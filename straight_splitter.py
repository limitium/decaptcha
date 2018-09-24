import glob
import numpy as np
from scipy import ndimage
from itertools import groupby
import matplotlib.pyplot as plt
from scipy import misc

THRESHOLD_COLOR_LINE = 150

# files = glob.glob("./inline/*.jpg")

# vertical_lines_median = [0., 2., 11., 26.5, 16., 14., 9., 8., 7., 5.5, 5., 3.5,
#                          4., 3.5, 4., 2., 1., 1.5, 2.5, 3., 1., 0., 0., 0.,
#                          0., 0.]

vertical_lines_median = [0., 2., 11., 26.5, 16.,
                         14., 9., 9., 8., 8,
                         7., 8, 8, 6, 6,
                         10, 9, 9, 8, 4.,
                         2., 1., 1., 1., 1.,
                         1.]
vertical_lines_median = [0, 37, 25, 20, 11, 8, 6, 5, 5, 4, 2, 1, 1]


def is_straight(img):
    y_max_lines_len = len(vertical_lines_median)
    vertical_lines = np.zeros(y_max_lines_len, dtype=float)
    binary_img = img < THRESHOLD_COLOR_LINE
    mask = binary_img.astype(int)
    for y_line in mask.T:
        if 1 in y_line:
            max_line = max(sum(g) for v, g in groupby(y_line) if v)
            vertical_lines[max_line] += 1
    for y in range(y_max_lines_len - 1, 4, -1):
        if vertical_lines[y] >= vertical_lines_median[y]:
            return True
    # print vertical_lines_median
    # print vertical_lines
    return False

# i = 0
# for name in files:
#     vertical_lines = np.zeros(26, dtype=float)
#     img = ndimage.imread(name, flatten=False, mode='L')
#     if is_straight(img):
#         i+=1
#         misc.imsave(name.replace("inline", "stest"), img)
# print i
