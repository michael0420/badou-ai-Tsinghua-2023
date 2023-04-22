# -*- coding: utf-8 -*-
"""
@author: 3-温葳方-天津

彩色图像的灰度化(手动)
"""
import cv2
import numpy as np

image_read = cv2.imread("lenna.png")
height, width, channels = image_read.shape
image_gray = np.zeros(shape=(height, width), dtype=image_read.dtype)
for row_coordinate in range(height):
    for column_coordinate in range(width):
        current_pixel = image_read[row_coordinate, column_coordinate]
        blue_value = current_pixel[0]
        green_value = current_pixel[1]
        red_value = current_pixel[2]
        image_gray[row_coordinate, column_coordinate] = int(blue_value * 0.11 + green_value * 0.59 + red_value * 0.3)

print(image_gray)
cv2.imwrite("gray_manually.png", image_gray)
cv2.imshow("image show gray", image_gray)
cv2.waitKey()
cv2.destroyAllWindows()
exit()
