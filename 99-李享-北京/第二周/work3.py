#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
# 绘制直方图
plt.hist(dst.ravel(), 256)
plt.show()

# 显示原始灰度图像和经过直方图均衡化后的图像
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

# 彩色图像直方图均衡化

# 读取彩色图像
img = cv2.imread("lenna.png", 1)

# 显示原始彩色图像
cv2.imshow("src", img)

# 对彩色图像进行直方图均衡化，需要分解通道并对每一个通道进行均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# 合并经过直方图均衡化处理的每个通道
result = cv2.merge((bH, gH, rH))

# 显示经过直方图均衡化处理后的彩色图像
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)