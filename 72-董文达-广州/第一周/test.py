from skimage.color import rgb2gray
import numpy
import matplotlib.pyplot as plt
import cv2

# 手工实现灰度化
img = cv2.imread('lenna.png')
# print(img)
h, w = img.shape[:2]
img_gray = numpy.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print("image show gray：%s" % img_gray)
cv2.imshow('gray image', img_gray)
# cv2.waitKey(0)

# 原图
plt.subplot(221)
img = plt.imread('lenna.png')
plt.imshow(img)
print('-----------image lenna-----------')
print(img)


# 调接口实现灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
print(img_gray)
plt.imshow(img_gray, cmap='gray')


# # 二值化
img_binary = numpy.where(img_gray >= 0.5, 1, 0)
print("----------------image show binary-------------------")
print(img_binary)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
