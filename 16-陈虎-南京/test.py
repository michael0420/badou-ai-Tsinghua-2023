from skimage.color import rgb2gray, rgba2rgb
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 获得图像的的数字数组
img = cv2.imread('dog.jpg')
# 获得图像的长宽，矩阵的规模
h, w = img.shape[:2]
# 创造一个空白画板
img_gray = np.zeros([h, w], img.dtype)
# 获得数据，填充灰度数据
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
print(img_gray)
cv2.imshow('gray', img_gray)



img = cv2.imread('dog.jpg')
cv2.imshow('1', img)

# 灰度化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('++++++++++++++++++')
print(img_gray)
cv2.imshow('2', img_gray)

# 二值化
img_binary = np.where(img_gray >= 128, 255, 0)
print('----------------------------')
# print(img_binary)
img_binary = img_binary.astype("uint8")
print(img_binary)
cv2.imshow('3', img_binary)

key = cv2.waitKey(2000)


plt.subplot(221)
img = plt.imread('dog.jpg')
img_rgb = rgba2rgb(img)
plt.imshow(img_rgb)


img_gray = rgb2gray(img_rgb)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print('----------------------------')
print(img_binary)
plt.imshow(img_binary, cmap='gray')

plt.show()




