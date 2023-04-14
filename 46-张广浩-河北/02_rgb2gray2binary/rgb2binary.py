# -*- coding: utf-8 -*-
# @Author  : ZGH
# @Time    : 2023/4/10 15:09
# @File    : rgb2binary.py
# @Software: PyCharm


import cv2
import numpy as np

from rgb2gray import rgb2gray_self

def rgb2binary_self1(img ,threshold):
    assert img is not None , "I can not find the image!"
    #第一中方法先灰度化在遍历，
    gray = rgb2gray_self(img)
    binary = np.zeros_like(gray)
    w,h,_ = gray.shape
    #遍历图像，大于阈值设置为255，小于阈值设置0
    for i in range(w):
        for j in range(h):
            if gray[i,j] > threshold:
                binary[i,j] = 255
            else:
                binary[i,j] = 0
    return binary

def rgb2binary_self2(img,threshold):
    assert img is not None,"I can not find the image !"

    h,w,c = img.shape
    #这里不能少uint8
    binary = np.zeros((h,w,1),np.uint8)
    for i in range(h):
        for j in range(w):
            #使用对应权重相乘
            value = img[i, j, 0] * 0.3 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.11
            #利用3目表达式
            value = 255 if value >= 127 else 0
            binary[i,j] = value
    return binary

#调用接口实现
def rgb2binary(img):
    assert img is not None, "I can not find the image!"
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    thread, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #cv2.threshold返回的是两个值，一个阈值，一个处理后图像
    return binary

def main():
    img = cv2.imread("girl.png")
    cv2.imshow("rgb_image", img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #注意这里imshow显示的默认BGR通道，转化RGB再直接用imshow会变色
    # cv2.imshow("rgb_image", img)

    #使用方法2
    binary = rgb2binary_self2(img , 127)
    #使用方法1
    #binary = rgb2binary_self1(img, 127)
    #调用接口
    #binary = rgb2binary(img)
    cv2.imshow("binary_image",binary)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()