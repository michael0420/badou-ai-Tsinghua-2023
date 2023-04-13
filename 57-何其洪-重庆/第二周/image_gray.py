# -*- coding: utf-8 -*-

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2


def img2gray(img, func):
    """
    将图片转换为灰度图
    :param img: 图片对象
    :return: 转换后的灰度图
    """
    height, width = img.shape[:2]  # (512, 512, 3)
    # zeros创建并填充0     形状     数据类型
    img_gray = np.zeros([height, width], img.dtype)
    for i in range(height):
        for j in range(width):
            # opencv的像素颜色顺序为BGR
            pixel = img[i, j]
            img_gray[i, j] = func(pixel)
    return img_gray


if __name__ == '__main__':
    img = cv2.imread('images/lenna.png')
    print("原图：", img, img.shape)
    # 原图
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 转灰度图
    # 使用公式Gray = R * 0.3 + G * 0.59 + B * 0.11
    img_gray_float = img2gray(img, lambda pixel: int(pixel[0] * 0.11 + pixel[1] * 0.59 + pixel[2] * 0.3))
    print("公式转灰度图：", img_gray_float, img_gray_float.shape)
    # img_gray_int = img2gray(img, lambda pixel: int((pixel[0] * 11 + pixel[1] * 59 + pixel[2] * 30) / 100))
    # img_gray_shifting = img2gray(img, lambda pixel: int((pixel[0] * 28 + pixel[1] * 151 + pixel[2] * 76) >> 8))
    # img_gray_avg = img2gray(img, lambda pixel: int((pixel[0] + pixel[1] + pixel[2]) / 3))
    # 将整个窗口分为两行两列，当前位置为 1
    plt.subplot(222)
    # 展示转换的灰度图，由于opencv返回的结果是BGR，此处转换为RGB
    plt.imshow(cv2.cvtColor(img_gray_float, cv2.COLOR_BGR2RGB))

    # 使用封装的api转灰度图
    plt.subplot(223)
    img_gray = rgb2gray(img)
    print("api转灰度图：", img_gray, img_gray.shape)
    plt.imshow(img_gray, cmap='gray')

    # 图片二值化
    plt.subplot(224)
    img_binary = np.where(img_gray >= 0.5, 1, 0)
    print("二值化：", img_binary, img_binary.shape)
    plt.imshow(img_binary, cmap='gray')

    plt.show()
