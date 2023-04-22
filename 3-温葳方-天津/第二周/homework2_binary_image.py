# -*- coding: utf-8 -*-
"""
@author: 3-温葳方-天津

彩色图像的二值化
"""
import cv2
import numpy as np

image_read = cv2.imread("lenna.png")
image_gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
image_binary = np.where(image_gray > 127, 255, 0)
image_binary = image_binary.astype(np.uint8)

cv2.imwrite("binary.png", image_binary)
cv2.imshow("binary", image_binary)
cv2.waitKey()
cv2.destroyAllWindows()
exit()
