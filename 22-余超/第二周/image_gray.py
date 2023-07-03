from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

im = Image.open('lenna.jpg')
img = np.asarray(im)
h,w =img.shape[:2]
# 读取img的前两个维度的长度
img_gray = np.zeros((h,w),img.dtype)

for i in range(h):
    for j in range(w):
        m = img[i,j]
# 注意此处m的数据结构，由于img是一个3维数组所以m是一个一维的数组
        img_gray[i,j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)
print("image show gray:%s"%img_gray)
plt.subplot(221)
plt.imshow(img)
img_gray1=rgb2gray(img)
# 把原图、灰度化的图片、二值化的图片放到一起做对比
plt.subplot(222)
plt.imshow(img_gray1,cmap='gray')
img_binary = np.where(img_gray >= 100, 255, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
