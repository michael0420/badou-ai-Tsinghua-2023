# -*- coding: utf-8 -*-
# @Author  : ZGH
# @Time    : 2023/4/9 17:08
# @File    : rgb2gray.py
# @Software: PyCharm

import cv2
import numpy as np


#自己手写实现，遍历每个像素点，记录其三个通道的加权值，赋予新图像像素
def rgb2gray_self(img):
    #assert写法，确保 img不为空 所以写到的时候就 is not none 保障不为空
    assert img is not None,"I can not find the image !"

    h,w,c = img.shape
    #这里不能少uint8
    gray = np.zeros((h,w,1),np.uint8)
    for i in range(h):
        for j in range(w):
            #使用对应权重相乘
            value = img[i, j, 0] * 0.3 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.11
            #使用均值
            # r,g,b = img[i,j]
            # value = (int(r)+int(g)+int(b))/3
            gray[i,j] = np.uint(value)
    return gray

#调用接口实现
def rgb2gray(img):
    assert img is not None, "I can not find the image !"
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return gray




def main():
    img = cv2.imread("girl.png")
    cv2.imshow("girl",img)
    #BGR转换RGB 注意写法
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = rgb2gray_self(img)
    cv2.imshow("gray_girl", gray)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
