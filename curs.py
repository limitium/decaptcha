from scipy import misc
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import glob

import regions
import line
import straight_splitter
from skimage.transform import rescale

BACKGROUND_COLOR = 222
# files = glob.glob("./imgs/*.jpg")
# labels_size = np.zeros(20, dtype=int)
# for name in files:
#     img = ndimage.imread(name, flatten=False, mode='L')
# img = ndimage.imread('imgs/0A4Xgl79z7.jpg', flatten=False, mode='L')

img = ndimage.imread('imgs/1Mj56110gn.jpg', flatten=False, mode='L')

# img = ndimage.imread('imgs/67sMWm6j1b.jpg', flatten=False, mode='L')


# img = ndimage.imread('imgs/aKFT9TAiK5.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/ABvgWcp14P.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/2fAuDc4e86.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/304iaqYHNN.jpg', flatten=False, mode='L')
# img = ndimage.im
#
# read('imgs/x4E4411udt.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/2JmM93Z83X.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/3CD73H94oM.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/2JmM93Z83X.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/02h680qXyK.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/b8p5n9iulS.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/4TGJpeLWWp.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/5elsG2noZ2.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/zWSx69142Z.jpg', flatten=False, mode='L')

# img = ndimage.imread('imgs/00BiyijyRP.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/2Up18CKkcQ.jpg', flatten=False, mode='L')
img = ndimage.imread('imgs/bkY7qXk2QW.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/Bw3XsZM7tf.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/bG27426eSy.jpg', flatten=False, mode='L')


# img = ndimage.imread('imgs/1evuLTXDjK.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/gHL4QNEDTK.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/2Bq95gD1rW.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/2DSnHzCM8g.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/2F0nKP4sH9.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/2sZ1kMxc31.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/1njg682yVu.jpg', flatten=False, mode='L')


# img = ndimage.imread('imgs/TIYTT9peZ1.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/zpdn4YBIKK.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/wVUtZKfNwD.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/0A54a5RSe0.jpg', flatten=False, mode='L')

# Check if image is straight
is_straight = straight_splitter.is_straight(img)

# Resize 2x
img = ndimage.interpolation.zoom(img, 2,
                                 order=1,
                                 mode='reflect')

# to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').

width = img.shape[1]
height = img.shape[0]


# Image to points over threshold
def points_over_threshold(image, threshold):
    mask = image < threshold
    points = []
    for x in range(0, width):
        for y in np.nonzero(mask[..., x])[0]:
            points.append([x, y])
    return np.array(points)


points = points_over_threshold(img, 200)

# Calculate 4 level curve over
polyVec = np.polyfit(points[:, 0], points[:, 1], 4)
poli = np.poly1d(polyVec)

xp = np.linspace(0, width - 1, width)
yp = poli(xp)

# Y offsets by from curve
offsets = np.round(yp - height / 2).astype(int)

# Draw regresion curve
plt.plot(points[:, 0], points[:, 1], '+', xp, yp, '-')

# Inline image by offsets
imgT = img.T
for x in range(len(offsets)):
    offset = offsets[x]
    offset_val = height if abs(offset) > height  else abs(offset)
    if offset > 0:
        zeros = np.ones(offset_val, dtype=int) * BACKGROUND_COLOR
        imgT[x] = np.concatenate((imgT[x][offset_val:], zeros), axis=0)
    elif offset < 0:
        np_zeros = np.ones(offset_val, dtype=int) * BACKGROUND_COLOR
        imgT[x] = np.concatenate((np_zeros, imgT[x][:-offset_val]), axis=0)

# Draw uncurved mask
mask = img < 120
plt.imshow(mask, cmap="Greys")


def regression_try_unital():
    global points, polyVec, poli, xp, yp
    plt.plot(points[:, 0], points[:, 1], '+', xp, yp, '-')
    points = points_over_threshold(img, 120)
    polyVec = np.polyfit(points[:, 1], points[:, 0], 1)
    poli = np.poly1d(polyVec)
    #
    xp = np.linspace(0, height - 1, height)
    yp = poli(xp)
    # plt.plot(points[:, 0], points[:, 1], '.')
    plt.plot(points[:, 1], points[:, 0], '.', xp, yp, '*')
    plt.plot(yp, xp, '--')


# regression_try_unital()
def draw_y_lines_histogramm():
    global x
    plt.imshow(img, cmap="Greys")
    mask = img < 120
    y_hist = [sum(y) for y in mask.T]
    # print y_hist
    plt.imshow(mask, cmap="Greys")
    for x in range(0, 200):
        if y_hist[x] > 0:
            plt.plot(np.full((1, y_hist[x]), x, dtype=int)[0], np.arange(0, y_hist[x], dtype=int))
    plt.hist(y_hist, 100, normed=1, facecolor='g', alpha=0.75)


# Y-lines histogramm
# draw_y_lines_histogramm()

# Unitallic image
if not is_straight:
    # print "straight"
    # else:
    # print "un italic"
    img = line.unitalic(img)

#
# from skimage.morphology import label
# from skimage.measure import regionprops
#
# label_image = label(img < 150, connectivity=2)
# regions = regionprops(label_image)
# labels_size[len(regions)] += 1
# if len(regions) < 8:
#     print name
#
# print regions
# for region in regionprops(label_image):
labeled_regions = regions.get_labeled_regions(img, plt)
print "total regions:",len(labeled_regions)
plt.show()
# print labels_size
