import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
#实现RGB2GRAY
#实现二值化

#手工灰度化
img = cv2.imread("lenna.jpg")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
#print(img_gray)
cv2.imshow("lenna1.jpg",img_gray)
cv2.waitKey(1000)
#cv2.imwrite("C:\...\yuhao\PycharmProjects\lenna1.png",img_gray,)

#调用库灰度化
img_gray=rgb2gray(img)
#plt.subplot(222)
cv2.imshow('lenna2.jpg',img_gray)
cv2.waitKey(1000)
#cv2.imwrite("C:\...\yuhao\PycharmProjects\lenna2.png",img_gray)

#二值化
h,w = img_gray.shape
img_binary = np.zeros([h,w])
for i in range(h):
    for j in range(w):
        if(img_gray[i,j] <= 0.5):
            img_binary[i,j] = 0
        else:
            img_binary[i,j] = 1
#print(img_binary)
#plt.subplot(223)
cv2.imshow('lenna3.jpg',img_binary)
cv2.waitKey(0)
#plt.show()
#cv2.imwrite("C:\...\yuhao\PycharmProjects\lenna3.png",img_binary)