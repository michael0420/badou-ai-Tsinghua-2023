import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

img = cv2.imread('img/lenna.png')
print(img.shape, img.dtype,type(img))
cv2.imshow('img',img)
h, w = img.shape[:2]
print(img[1,2,2])

#灰度化
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print(img_gray.shape)
print(img_gray)
cv2.imshow('image show gray',img_gray)


#二值化
img_b = np.zeros([h, w], img.dtype)
rows,cols = img_gray.shape
# print(img_gray[2,2])
for i in range(rows):
    for j in range(cols):
        if(img_gray[i,j]<=127):
            img_b[i,j]=0
        else:
            img_b[i,j]=255
print(img_b.shape)
print(img_b)
cv2.imshow('image show bw',img_b)
cv2.waitKey(0)

#
plt.subplot(221)
img1 = plt.imread('img/lenna.png')
plt.imshow(img1)
print(img1)

img_gray1 = color.rgb2gray(img1)
plt.subplot(222)
plt.imshow(img_gray1,cmap='gray')
print(img_gray1)

img_binary1=np.where(img_gray1 >=0.5,1,0)
plt.subplot(223)
plt.imshow(img_binary1,cmap='gray')
print(img_binary1)

plt.subplot(224)
plt.imshow(img_gray1,cmap='bone')

plt.show()
