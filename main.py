import time
import cv2

from hough import hough_transform
from processing import preprocess_image, find_angle, rotate_image, draw_plots

times = []
image_size = []
square_original = []
angles = []

for name_file in range(1, 11):
    path = "data/" + str(name_file) + ".jpg"
    image = cv2.imread(path)
    image_preprocessed = preprocess_image(image)
    h, w = image_preprocessed.shape[:2]
    image_rot = cv2.rotate(image_preprocessed, cv2.ROTATE_90_CLOCKWISE)

    start = time.time()

    transformed_1 = hough_transform(image_preprocessed, 0, w, True)
    transformed_2 = hough_transform(image_preprocessed, 0, w, False)
    transformed_3 = hough_transform(image_rot, 0, h, True)
    transformed_4 = hough_transform(image_rot, 0, h, False)

    end = time.time()

    times.append(1000 * (end - start))
    image_size.append(image_preprocessed.shape[0] * image_preprocessed.shape[1] / 1000000)

    angle, angle_sign = find_angle(transformed_1, transformed_2, transformed_3, transformed_4)

    if angle_sign == 1 or angle_sign == 3:
        angle *= -1

    angles.append(-angle)

    linear_trans = rotate_image(image, angle, method=cv2.INTER_LINEAR)
    cv2.imwrite('results/' + str(name_file) + '_linear.jpg', linear_trans)

    nearest_trans = rotate_image(image, angle, method=cv2.INTER_NEAREST)
    cv2.imwrite('results/' + str(name_file) + '_nearest.jpg', nearest_trans)

draw_plots(image_size, times)

with open('results/angles.txt', 'w') as f:
    for i, angle in enumerate(angles):
        f.write(f"{i}) {angle}\n")