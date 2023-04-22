# -*- coding: utf-8 -*-
"""
@author: 3-温葳方-天津

彩色图像的灰度化(调用接口)
"""
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray

# plt.subplot(2, 2, 1)
image_read = cv2.imread("lenna.png")
# image_read = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
# image_gray = rgb2gray(image_read)
image_gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_api.png", image_gray)

cv2.imshow("gray", image_gray)
cv2.waitKey()
cv2.destroyAllWindows()
exit()
