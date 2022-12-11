import numpy as np


def hough_transform(img, left_border, right_border, pos=True):
    h, w = img.shape[:2]

    res = np.zeros([h, right_border - left_border])

    if right_border - left_border == 1:
        res[:, 0] = img[:, left_border]
        return res

    left_transformed = hough_transform(img, left_border, (left_border + right_border) // 2, pos)
    right_transformed = hough_transform(img, (left_border + right_border) // 2, right_border, pos)

    for i in range(h):
        for j in range(right_border - left_border):
            if pos:
                res[i, j] = left_transformed[i, j // 2] + right_transformed[(i + j // 2 + j % 2) % h, j // 2]
            else:
                res[i, j] = left_transformed[i, j // 2] + right_transformed[(i - j // 2 - j % 2) % h, j // 2]

    return res
