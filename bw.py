from scipy import misc
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# img = ndimage.imread('imgs/0A4Xgl79z7.jpg', flatten=False, mode='L')
# img = ndimage.imread('imgs/1Mj56110gn.jpg', flatten=False, mode='L')
img = ndimage.imread('imgs/67sMWm6j1b.jpg', flatten=False, mode='L')
img = ndimage.imread('imgs/aKFT9TAiK5.jpg', flatten=False, mode='L')
img = ndimage.imread('imgs/ABvgWcp14P.jpg', flatten=False, mode='L')
img = ndimage.imread('imgs/2fAuDc4e86.jpg', flatten=False, mode='L')
img = ndimage.imread('imgs/304iaqYHNN.jpg', flatten=False, mode='L')
img = ndimage.imread('imgs/x4E4411udt.jpg', flatten=False, mode='L')

print "type"
print type(img)
print "shape,dtype"
print img.ndim, img.shape, img.dtype
print "mean"
print img.mean()
# print "img 20"
# print img[20]

# hist, bin_edges = np.histogram(img, bins=10)
# bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# print "hist"
# print hist
# print "bin centers"
# print bin_centers
binary_img = img < 180
# binary_img = img <= img.mean()
# mask = img > (img.mean()-20)
# label_im, nb_labels = ndimage.label(mask)
# print label_im[25]
# plt.imshow(label_im)
# plt.imshow(img)
# plt.plot(bin_centers, hist, lw=2)
# print "binary 20"
# print binary_img[20]
# already binary.
mask = binary_img.astype(int)
print "mask full"
print mask
points = []
for x in range(0, 100):
    for y in np.nonzero(mask[..., x])[0]:
        points.append([x, y])
print '======'
# print mask.T.shape
# print mask.T[50]

points = np.array(points)
# print np.array(points)[:,0]
polyVec = np.polyfit(points[:, 0], points[:, 1], 4)
poli = np.poly1d(polyVec)

xp = np.linspace(0, 99, 100)
yp = poli(xp)
# offsets = yp.astype(int) - img.shape[0] / 2
offsets = np.round(yp - img.shape[0] / 2).astype(int)

print offsets
maskT = mask.T
# for x in range(len(offsets)):
#     if offsets[x] > 0:
#         maskT[x] = np.concatenate((maskT[x][offsets[x]:], np.zeros(offsets[x]).astype(int)), axis=0)
#     elif offsets[x] < 0:
#         maskT[x] = np.concatenate((np.zeros(abs(offsets[x])).astype(int), maskT[x][:offsets[x]]), axis=0)
imgT = img.T
for x in range(len(offsets)):
    if offsets[x] > 0:
        zeros = np.zeros(offsets[x], dtype=int)
        zeros.fill(222)
        imgT[x] = np.concatenate((imgT[x][offsets[x]:], zeros.astype(int)), axis=0)
    elif offsets[x] < 0:
        np_zeros = np.zeros(abs(offsets[x]), dtype=int)
        np_zeros.fill(222)
        imgT[x] = np.concatenate((np_zeros, imgT[x][:offsets[x]]), axis=0)

plt.plot(points[:, 0], points[:, 1], '+', xp, yp, '-')
# plt.imshow(mask, cmap="Greys")
plt.imshow(imgT.T, cmap="Greys")

plt.show()

# np.polysub()

# print (x for x in mask[...,c]c )
# print "mask x50"
# print mask[...,50]

# plt.imshow(img)
# np.polyfit()

# plt.show()
# from sklearn.feature_extraction import image
# from sklearn.cluster import spectral_clustering
#
# graph = image.img_to_graph(img, mask=mask)
# graph.data = np.exp(-graph.data / graph.data.std())
# labels = spectral_clustering(graph, n_clusters=1, eigen_solver='arpack')
# label_im = -np.ones(mask.shape)
# label_im[mask] = labels
#
# from skimage import measure
# all_labels = measure.label(binary_img)
# print all_labels[20]
# plt.imshow(all_labels)
# plt.show()
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()
