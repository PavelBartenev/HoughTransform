import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


def preprocess_image(image):
    image = cv.cvtColor(cv.GaussianBlur(image, (3, 3), 0), cv.COLOR_BGR2GRAY)

    x_sobel = cv.Sobel(image, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    y_sobel = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    image = cv.addWeighted(cv.convertScaleAbs(x_sobel), 0.5,
                           cv.convertScaleAbs(y_sobel), 0.5, 0)

    h, w = image.shape

    new_h = 2**(int(np.log2(h)) + 1)
    new_w = 2**(int(np.log2(w)) + 1)

    new_image = np.zeros([new_h, new_w])
    new_image[0:h, 0:w] = image

    return new_image


def find_angle(*res):
    res_array = list(res)

    cur_max = 0
    cur_pos = 0
    n = 0
    index = 0

    for i in range(len(res_array)):
        for j in range(res_array[i].shape[1]):
            cur_sum = np.sum(res_array[i][:,j] ** 2)

            if cur_sum > cur_max:
                cur_pos = j
                n=res_array[i].shape[1]
                index = i
                cur_max = cur_sum

    angle = 45 / n * cur_pos
    index_angle = index

    return angle, index_angle


def rotate_image(img, angle, method):
    h, w = img.shape[:2]

    matrix = cv.getRotationMatrix2D((w/2, h/2), angle, 1)

    sin_rotation = math.sin(math.radians(angle))
    cos_rotation = math.cos(math.radians(angle))

    w_border = int((h * abs(sin_rotation)) + (w * abs(cos_rotation)))
    h_border = int((h * abs(cos_rotation)) + (w * abs(sin_rotation)))

    matrix[0, 2] += ((w_border / 2) - w/2)
    matrix[1, 2] += ((h_border / 2) - h/2)

    image = cv.warpAffine(img, matrix, (w_border, h_border), flags=method)

    return image


def draw_plots(image_size, times):
    megapixel_times = []

    for i in range(len(image_size)):
        megapixel_times.append([image_size[i], times[i]])
    megapixel_times.sort()

    res = [list(megapixel_times[0])]

    for x in megapixel_times[1:]:
        if res[-1][0] == x[0]:
            res[-1][1] = (res[-1][1] + x[1]) / 2
        else:
            res.append(list(x))

    res = np.array(res)
    plt.plot(res[:, 0], res[:, 1], color='blue')
    plt.xlabel("Размер картинки, Mp")
    plt.ylabel("Время работы, msec")
    plt.savefig('results/time_plot.png')
