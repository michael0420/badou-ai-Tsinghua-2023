from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("E:\AAAAAA\FIRST.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)


plt.subplot(221)
img = plt.imread("E:\AAAAAA\FIRST.png")
img = cv2.imread("E:\AAAAAA\FIRST.png", False)
plt.imshow(img)
print("---image first----")
print(img)

img_gray = rgb2gray(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----image_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
