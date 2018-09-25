import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.measure._regionprops import _RegionProperties
from skimage.morphology import label
from skimage.morphology import watershed


def is_small(region):
    minh, minw, maxh, maxw = region.bbox
    width = maxw - minw
    height = maxh - minh
    return height < 15


def is_horizontal(region):
    minh, minw, maxh, maxw = region.bbox
    width = maxw - minw
    height = maxh - minh
    return width > height


def is_wide(region):
    minh, minw, maxh, maxw = region.bbox
    width = maxw - minw
    height = maxh - minh
    return width > 24


def merge(region1, region2, image):
    minh1, minw1, maxh1, maxw1 = region1.bbox
    minh2, minw2, maxh2, maxw2 = region2.bbox

    minh = min(minh1, minh2)
    minw = min(minw1, minw2)
    maxh = max(maxh1, maxh2)
    maxw = max(maxw1, maxw2)

    return create_new_region(image, maxh, maxw, minh, minw, region1.label + region2.label)


def create_new_region(image, maxh, maxw, minh, minw, title):
    return _RegionProperties((slice(minh, maxh), slice(minw, maxw)), title, image, image, False,'xy')


def region_center(r):
    minh, minw, maxh, maxw = r.bbox
    return minw + (maxw - minw) / 2


def nearest(region, regions):
    nrst = region
    ndist = 9999999
    for r in regions:
        dist = abs(region_center(region) - region_center(r))
        if dist < ndist:
            ndist = dist
            nrst = r
    return nrst


def split(region, image):
    minh, minw, maxh, maxw = region.bbox
    min_col_position = 0
    min_col_value = 999
    # 15 from is_small
    for col in np.arange(minw + 6, maxw - 6):
        cur_col_value = 0
        for row in np.arange(minh, maxh):
            cur_col_value = cur_col_value + image[row, col]
        if 0 < cur_col_value < min_col_value:
            min_col_value = cur_col_value
            min_col_position = col
    print "End min col", min_col_position, "val", min_col_value
    left_region = create_new_region(image, maxh, min_col_position, minh, minw, region.label + 2)
    right_region = create_new_region(image, maxh, maxw, minh, min_col_position, region.label + 1)
    splitted = []
    # if not is_small(left_region):
    splitted.append(left_region)
    # if not is_small(right_region):
    splitted.append(right_region)
    return splitted


def regionWidthSort(r):
    minh, minw, maxh, maxw = r.bbox
    return maxw - minw


def optimize(raw_regions, img):
    optimized = raw_regions[:]
    toOptimize = raw_regions[:]
    i = 0
    for region in toOptimize:
        i += 1
        minh, minw, maxh, maxw = region.bbox
        width = maxw - minw
        height = maxh - minh
        print i, '- h:', height, 'w:', width
        if is_wide(region):
            print 'huge'
            hudges = [region]
            while len(hudges) > 0:
                huge = hudges.pop()
                if huge in optimized:
                    optimized.remove(huge)
                for splited in split(huge, img):
                    if is_wide(splited):
                        hudges.append(splited)
                    else:
                        optimized.append(splited)
    i = 0
    for region in toOptimize:
        i += 1
        minh, minw, maxh, maxw = region.bbox
        width = maxw - minw
        height = maxh - minh
        print i, '- h:', height, 'w:', width
        if is_small(region):
            optimized.remove(region)
            print 'small, looking for nearest'
            nrst = nearest(region, optimized)
            optimized.remove(nrst)
            print 'merge small to nearest'
            merged = merge(region, nrst, img)
            optimized.append(merged)
    return optimized


def get_labeled_regions(img, plt):
    # Calcualte regions

    image = img < 160

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    distance = image
    local_maxi = peak_local_max(distance,
                                indices=False,
                                footprint=np.ones((3, 3)),
                                labels=image)
    # print local_maxi
    markers = ndi.label(local_maxi, structure=[[1, 1, 1],
                                               [1, 1, 1],
                                               [1, 1, 1]])[0]
    labels = watershed(~distance, markers, mask=image)

    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(9, 9), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    #
    ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(~distance, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.gray, interpolation='nearest')
    ax[2].set_title('Separated objects')

    # shoftimg = ndimage.interpolation.shift(img, [0, -15], cval=BACKGROUND_COLOR)
    # rotimg = ndimage.interpolation.affine_transform(
    #     shoftimg, [[1, 0], [-0.4, 1.0]], cval=BACKGROUND_COLOR
    # )
    rotimg = img
    ax[3].imshow(rotimg < 180, cmap=plt.cm.gray)
    ax[3].set_title('Rotated')

    import skimage.filters as filters
    from skimage import feature

    bw = filters.threshold_local(image, 9)

    ax[4].imshow(bw, cmap=plt.cm.gray)
    ax[4].set_title('Adaptive threshold')

    edges = feature.canny(bw, sigma=1, low_threshold=0)
    ax[5].imshow(edges, cmap=plt.cm.gray)
    ax[5].set_title('edges')

    # print edges
    # label_image = label()
    label_image = label(edges)
    rotimg_ = rotimg < 170
    image = rotimg_
    # label_image = label(rotimg_, connectivity=img.ndim)

    print "image dimension:", img.ndim
    label_image = label(image, connectivity=2)

    ax[6].imshow(image, cmap=plt.cm.gray)
    ax[6].set_title('Labeled items')
    ax[6].axis('off')
    regions = regionprops(label_image)

    draw_region_square(ax[6], regions, 'red')
    regions = optimize(regions, label_image)

    draw_region_square(ax[6], regions, 'green')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()

    return regions


# def get_labeled_regions(img):
#     image = img < 150
#
#     label_image = label(image, connectivity=2)
#
#     regions = regionprops(label_image)
#
#     return optimize(regions, label_image)


def draw_region_square(plot, regions, color='red'):
    for region in regions:
        # Draw rectangle around segmented coins.
        minh, minw, maxh, maxw = region.bbox
        print "Region: h1=", minh, "w1=", minw, "h2=", maxh, "w2=", maxw
        rect = mpatches.Rectangle((minw, minh),
                                  maxw - minw,
                                  maxh - minh,
                                  fill=False,
                                  edgecolor=color,
                                  linewidth=1)
        plot.add_patch(rect)
