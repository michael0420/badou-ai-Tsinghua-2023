import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
#手工实现
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
#print(img_gray)
cv2.imshow("img gray", img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

#调接口实现
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

#实现二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()


