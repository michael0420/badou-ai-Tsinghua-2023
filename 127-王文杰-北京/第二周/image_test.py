# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np

#原图
img = cv2.imread("lenna.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(2,3,1)
plt.imshow(img)

# 手工实现灰度化
#获取图片的high和wide
h,w = img.shape[:2]
#创建一张和当前图片大小一样的单通道图片
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        # 取出当前high和wide中的RGB
        m = img[i,j]
        # 将RGB转化为gray并赋值给新图像
        img_gray[i,j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)
plt.subplot(2,3,2)
plt.imshow(img_gray, cmap='gray')

# 手工实现二值化
img_binary = np.zeros([h,w],img_gray.dtype)
rows, cols = img_binary.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 127):
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255
plt.subplot(2,3,3)
plt.imshow(img_binary, cmap='gray')

#原图
plt.subplot(2,3,4)
plt.imshow(img)

#接口实现灰度化
img_gray_api = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(2,3,5)
plt.imshow(img_gray_api, cmap='gray')

#接口实现二值化
ret, img_binary_api = cv2.threshold(img_gray_api, 127, 255, cv2.THRESH_BINARY)
plt.subplot(2,3,6)
plt.imshow(img_binary_api, cmap='gray')

plt.show()