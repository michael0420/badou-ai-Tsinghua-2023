"""
@author:songxinran

彩色图像灰度化-手写实现
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")

h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

img = plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.show()



