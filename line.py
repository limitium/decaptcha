import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


# img = ndimage.imread('inline/1evuLTXDjK.jpg', flatten=False, mode='L')
# img = ndimage.imread('inline/gHL4QNEDTK.jpg', flatten=False, mode='L')
# img = ndimage.imread('inline/2Bq95gD1rW.jpg', flatten=False, mode='L')
# img = ndimage.imread('inline/2DSnHzCM8g.jpg', flatten=False, mode='L')
# img = ndimage.imread('inline/2F0nKP4sH9.jpg', flatten=False, mode='L')
# img = ndimage.imread('inline/2sZ1kMxc31.jpg', flatten=False, mode='L')
# img = ndimage.imread('inline/1njg682yVu.jpg', flatten=False, mode='L')
# img = ndimage.interpolation.zoom(img, 2, order=1, mode='reflect')

# im_array = np.array(img)
# plt.imshow(im_array, cmap="Greys")
# plt.show()

def unitalic(img):
    width = img.shape[1]
    height = img.shape[0]

    pixels_array = img < 120

    # plt.imshow(pixels_array, cmap="Greys")

    a = 1
    b = 0
    # line_x = np.arange(width)
    # line_y = a * line_x + b

    points = []

    tmp_len = 0
    step = 1

    while a < 5:
        begin = -width
        end = begin + width
        lines_array = []
        while begin < width:
            points_array = []
            for x in range(0, width, 1):
                y = int(round(a * x + b))
                if 0 <= y < height:
                    # if pixels_array[y][x]:
                    if not pixels_array[y][x]:
                        points_array.append([x, y])
                    else:
                        points_array = []
                        break
            begin += step
            end += step
            if len(points_array) > 0:
                lines_array.append(points_array)
        a += 0.2
        line_len = len(lines_array)
        if line_len > tmp_len:
            tmp_len = line_len
            # points.append(lines_array)
            break

    # for point in points:
    #     for lines_array in point:
    #         x = []
    #         y = []
    #         for line in lines_array:
    #             x.append(line[0])
    #             y.append(-line[1] + height)
    #         plt.plot(x, y)

    # print a

    # for y in range(0, height, 1):
    #     x = int(round(y / a))
    #     if x > 0:
    #         img[y] = np.concatenate((np.zeros(x), img[y][:-x]), axis=0)

    # img = ndimage.interpolation.shift(img, [0, -15], cval=222)
    # img = ndimage.interpolation.affine_transform(
    #     img, [[1, 0], [-0.3, 1.0]], cval=222
    # )
    # print a,"asd"
    for y in range(0, height, 1):
        x = int(round(y / a))
        if x > 0:
            img[height - y - 1] = np.concatenate((img[height - y - 1][x:], np.full(x, 222)), axis=0)

    return img

# mask = img < 120
# plt.imshow(mask, cmap="Greys")

# plt.show()
