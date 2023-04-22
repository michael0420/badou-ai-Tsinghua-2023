# -*- coding: utf-8 -*-
"""
@author: 81-于海洋
第二周 彩色图像的灰度化、二值化

TODO：
[0-255] [0-1] 都可以
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

IMAGE_NAME = "lenna.png"


def show_img(img, index: int, is_gray: bool = True, title: str = ''):
    plt.subplot(2, 4, index)
    if title != '' and title is not None:
        plt.title(title)

    if is_gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img, cmap='gray')


def show_origin():
    show_img(img_cv, 1, is_gray=False, title="CV")
    show_img(img_plt, 5, is_gray=False, title="PLT")


def show_cv():
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    show_img(img_gray, 2, title="CV2Gray")

    img_binary = np.where(img_gray > 128, 255, 0)
    show_img(img_binary, 6, title="CV2Binary")


def show_plt():
    img_gray = rgb2gray(img_plt)
    show_img(img_gray, 3, title="PLT2Gray")

    img_binary = np.where(img_gray > 0.5, 1, 0)
    show_img(img_binary, 7, title="PLT2Binary")


def show_manual():
    # shape[0] 图像垂直高度
    # shape[1] 图像水平宽度
    # shape[3] 图像通道数
    h, w, c = img_cv.shape[:3]
    print("c:", c)
    img_gray = np.zeros([h, w], img_cv.dtype)
    for i in range(h):
        for j in range(w):
            p = img_cv[i][j]
            img_gray[i][j] = p[0] * 0.3 + p[1] * 0.59 + p[2] * 0.11

    show_img(img_gray, 4, title="M2Gray")
    img_binary = np.where(img_gray > 128, 255, 0)
    show_img(img_binary, 8, title="M2Binary")


if __name__ == '__main__':
    img_cv = cv2.imread(IMAGE_NAME)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    img_plt = plt.imread(IMAGE_NAME)
    show_origin()
    show_cv()
    show_plt()
    show_manual()
    plt.show()
