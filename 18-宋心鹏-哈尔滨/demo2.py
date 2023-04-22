"""手写实现RGB图片灰度化"""
import numpy as np
import cv2

img = cv2.imread("lenna.png")
print(img)#三维数组仅有BGR
print(img.shape)#(512,512,3)高宽通道
h,w = img.shape[:2]#高宽传给hw
img_gray = np.zeros([h,w],img.dtype)#创建与图片大小相等全零数组
print(img_gray)
for i in range(h):
    for j in range(w):
        m = img[i,j]                             #取出当前high和wide中的BGR坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)#浮点算法
print(img_gray)
print(img_gray.shape)#512*512
cv2.imshow("img-gary",img_gray)
cv2.waitKey(0)#等待鼠标结束