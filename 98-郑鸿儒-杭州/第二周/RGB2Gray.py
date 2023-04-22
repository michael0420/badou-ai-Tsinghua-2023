import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
import cv2


img = cv2.imread('lenna.png')
h, w = img.shape[: 2]
# 灰度 手动
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        cur = img[i, j]
        img_gray[i, j] = int(cur[0] * 0.11 + cur[1] * 0.59 + cur[2] * 0.3)

print(img_gray)
cv2.imshow("rgb to gray", img_gray)
# imshow 自动闪退
cv2.waitKey()

# 灰度 接口1
img_gray = rgb2gray(img)
cv2.imshow("rgb to gray", img_gray)
# imshow 自动闪退
cv2.waitKey()

# 灰度 接口2
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("rgb to gray", img_gray)
# imshow 自动闪退
cv2.waitKey()

# 二值 手动
img_binary = np.zeros([h, w], img.dtype)
for m in range(h):
    for n in range(w):
        if img_gray[m, n] / 255 > 0.5:
            img_binary[m, n] = 1
plt.subplot(221)
plt.imshow(img_binary, cmap='gray')

# 二值 接口
img_binary = np.where(img_gray / 255 < 0.5, 0, 1)
plt.subplot(222)
plt.imshow(img_binary, cmap='gray')
plt.show()

