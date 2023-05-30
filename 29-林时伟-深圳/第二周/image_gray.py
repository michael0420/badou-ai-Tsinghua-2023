# -*- coding: utf-8 -*-
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化
plt.subplot(211)
img = cv2.imread("resources/lenna.png")
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w])  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = float(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3) # 将BGR坐标转化为gray坐标并赋值给新图像
plt.imshow(img_gray, cmap='gray')

# 二值化
img_gray = rgb2gray(img)
img_binary = np.zeros([h, w])  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        if img_gray[i, j] <= 0.5:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 1
plt.subplot(212)
plt.imshow(img_binary, cmap='gray')
plt.show()
