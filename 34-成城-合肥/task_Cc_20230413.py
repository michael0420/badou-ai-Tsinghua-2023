#coding: utf-8

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread(r"C:\Users\57891\Documents\Baiduwangpan\lenna.png") #导入图片数据
h,w = img.shape[:2] #确定图片长宽
print (img.shape)

##灰度化手工实现
imp_1 = np.empty((h, w),img.dtype) #申明一个空图片结构
for i in range(h):
    for j in range(w):
        m = img[i, j] #读取图片数据
        imp_1[i, j] = m[0]*0.11+ m[1]*0.59+ m[2]*0.3 #灰度化图片并填充进空图片
print (imp_1)
cv2.imshow("image show gray",imp_1)

##二值化手工实现
h1, w1 = imp_1.shape[:2]
imp_2 = np.empty((h1, w1),img.dtype)
for i in range(h1):
    for j in range(w1):
        if (imp_1[i, j] <= 122): #二值化判断
            imp_2[i, j] = 0
        else:
            imp_2[i, j] = 1
print (imp_2)
cv2.imshow("image show erzhi",imp_2)

plt.subplot(221)
plt.title('lenna')
plt.imshow(img1) #原图呈现

plt.subplot(222)
plt.title('lenna_gray')
plt.imshow(imp_1, cmap='gray') #灰度图呈现

plt.subplot(223)
plt.title('lenna_2zhihua')
plt.imshow(imp_2, cmap='gray') #二值图呈现

plt.subplots_adjust(hspace=0.5) #调整子图纵向间距
plt.show()


# 灰度化调接口实现
img_gray = rgb2gray(img1)
plt.subplot(221)
plt.imshow(img1)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')