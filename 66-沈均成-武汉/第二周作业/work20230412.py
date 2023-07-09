# import cv2
# import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


# 1.实现RGB2GRAY

# 手工实现
# img = cv2.imread("lenna.png")
# h, w = img.shape[:2]                  # 获取图片lenna的high和wide
# img_gray = np.zeros([h, w], img.dtype)  # 创建一张和lenna一样的空白图片
# for i in range(h):
#     for j in range(w):
#         m = img[i, j]
#         img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   # 利用公式将rgb坐标灰值化
# print(h, w)
# print(img_gray)
# print(img)
# cv2.imshow("image show gray", img_gray)

plt.subplot(2, 2, 1)
img = plt.imread("lenna.png")
plt.imshow(img)

# 调用接口实现
img_gray = rgb2gray(img)
plt.subplot(2, 2, 2)
plt.imshow(img_gray, cmap='gray')

# 2.实现二值化
high, wide = img_gray.shape
for i in range(high):
    for j in range(wide):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
plt.subplot(2, 2, 3)
plt.imshow(img_gray, cmap='gray')

plt.show()