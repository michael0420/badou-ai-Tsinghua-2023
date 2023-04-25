# -*- coding: utf-8 -*-
"""
@author: 洪羽寒

彩色图像的灰度化、二值化
"""
import time
start_time = time.time()

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
hashmap={}
for i in range(h):
    hashmap[h] = i
    for j in range(w):
        hashmap[j] = j
        if j is not None and i:
            m = img[i,j]
            img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)


plt.subplot(221)
img = plt.imread("lenna.png") 

plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

end_time = time.time()
print("程序启动时间：", end_time - start_time, "秒")

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()

end_time = time.time()
print("程序终止时间：", end_time - start_time, "秒")
