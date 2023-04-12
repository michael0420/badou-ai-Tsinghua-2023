import unittest
import cv2
from matplotlib import pyplot as plt
import numpy as np

def rgb2gray(img):
    # 获取图像的高、宽和通道数
    h, w, c = img.shape
    # 创建一张和当前图像大小一样的单通道图像
    img_gray = np.zeros([h, w], dtype=np.uint8)
    # 对图像进行遍历，计算每个像素的灰度值
    for i in range(h):
        for j in range(w):
            # 获取当前像素的RGB值
            r, g, b = img[i, j]
            # 根据公式计算灰度值并赋值给新图像
            gray = int(0.3 * r + 0.59 * g + 0.11 * b)
            img_gray[i, j] = gray
    return img_gray

def erzhihua(img):
    # 获取图像的高、宽和通道数
    h, w, c = img.shape
    # 创建一张和当前图像大小一样的单通道图像
    img_gray = np.zeros([h, w], dtype=np.uint8)
    # 对图像进行遍历，计算每个像素的灰度值
    for i in range(h):
        for j in range(w):
            # 获取当前像素的RGB值
            r, g, b = img[i, j]

            # 根据公式计算灰度值并赋值给新图像
            if (r+g+b)/3 <int(255/3/2):
                gray = int(1)
            else:
                gray = int(0)
            img_gray[i, j] = gray
    return img_gray




if __name__ == '__main__':



    img = cv2.imread('lenna.png')
    # 手工实现
    img_gray2 = rgb2gray(img)
    plt.subplot(221)
    plt.imshow(img_gray2, cmap='gray')
    #调接口
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    plt.subplot(222)
    plt.imshow(img_gray, cmap='gray')

    img_gray3 = erzhihua(img)

    plt.subplot(223)
    plt.imshow(img_gray3, cmap='gray')

    plt.show()
