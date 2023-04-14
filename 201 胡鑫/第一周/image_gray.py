import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray



# 使用opencv读取图片并返回图片的矩阵（读取的图片的颜色->默认储存的格式为BGR）

# img = cv2.imread('lenna.png') # 第二个参数表示读取图片的形式默认为1：加载彩色图片；0：加载灰度图片；-1：包括alpha通道
# h, w = img.shape[:2]
# print(img[0, 1])
# print(h)
#
# # 灰度化（原理）
# # img_gray = np.zeros([h, w], dtype=img.dtype)
# # print(img_gray)
# # for i in range(h):
# #     for j in range(w):
# #         m = img[i, j]   # 取出每个像素中的BGR
# #         img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3) #BGR to Gray
#
# # 灰度化（调用函数）(cv2.cvtColor)
# img_gray = np.zeros([h, w], dtype=img.dtype)
# print(img_gray)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print('imgage show gray: %s' % img_gray)
# cv2.imshow('image show gray', img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# 使用matplotlib读取图片并返回图片的矩阵

plt.subplot(221)
img = plt.imread('lenna.png') # plt.imread: 如果读取的是rgb图像，shape返回（h, w, 3）
print(img.shape)
# print(img[0, 1])
plt.imshow(img)
print('---image lenna----')
print(img[0, 1])
print(img)

# 灰度化（调用skimage封装好的函数）
# 使用rgb2gray函数
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print('---image_gray lenna---')
print(img_gray[0, 1])
print(img_gray)

# 二值化（原理）
# print(img_gray.shape) # 灰度化后，shape返回（h, w）
h, w = img_gray.shape
for i in range(h):
    for j in range(w):
        if (img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
img_binary = np.where(img_gray > 0.5, 0, 1)
plt.subplot(223)
plt.imshow(img_binary, cmap='binary')
print('---image_binary lenna---')
print(img_binary[0, 1])
print(img_binary)
print(img_binary.shape)

plt.show()