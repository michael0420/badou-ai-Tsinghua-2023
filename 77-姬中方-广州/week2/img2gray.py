# -*- coding: utf-8 -*-
"""
@author: jizhongfang

1. 彩色图像灰度化，使用接口+手动实现
2. 图像二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 1. 手动实现度化

# 读取图片
img = cv2.imread("lenna.png")
# 表示将整个图像窗口分为2行2列, 当前位置为1
plt.subplot(221)
# BGR转RGB
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
# 获取高度和宽度
height,width = img.shape[:2]
# 创建一张和当前图片大小一样的单通道图片
img_gray = np.zeros([height,width],img.dtype)
# 遍历图片
for i in range(height):
    for j in range(width):
        # 取出当前高度和宽度中的RGB坐标
        m = img[i,j]
        # 将BGR坐标转化为gray坐标并赋值给新图像,计算公式Gray=R0.3+G0.59+B0.11
        img_gray[i,j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)
# 解决图片显示不全问题
#cv2.namedWindow('image show gray',cv2.WINDOW_NORMAL)
#cv2.imshow('image show gray',img_gray)
plt.subplot(222)
# cmap: 颜色图谱,默认绘制为RGB(A)颜色空间，gray 灰度显示
plt.imshow(img_gray, cmap='gray')

# 2.调用函数实现灰度化
img_gray2 = rgb2gray(img)
plt.subplot(223)
plt.imshow(img_gray2, cmap='gray')

# 3.二值化，将图像上的像素点的灰度值设置为0或1，整个图像呈现出明显的只有黑和白的视觉效果
img_binary = np.where(img_gray2 >= 0.5,1,0)
plt.subplot(224)
plt.imshow(img_binary, cmap='gray')

# 程序运行完，不闪退
plt.show()
#cv2.waitKey(0)
