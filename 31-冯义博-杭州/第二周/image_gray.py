from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


img = cv2.imread("lenna.png")


# 灰度化
# h, w = img.shape[:2]
# img_gray = np.zeros([h, w], img.dtype)
# for i in range(h):
#     for j in range(w):
#         p = img[i, j]
#         img_gray[i, j] = int(0.11 * p[0] + 0.59 * p[1] + 0.3 * p[2])
# cv2.imshow("image show gray", img_gray)
# cv2.waitKey(0)

# img_gray = rgb2gray(img)
# plt.subplot(221)
# plt.imshow(img_gray, cmap='gray')
# plt.show()


# 二值化
img_gray = rgb2gray(img)
h, w = img_gray.shape
for i in range(h):
    for j in range(w):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
cv2.imshow("image show binary", img_gray)
cv2.waitKey(0)

# img_gray = rgb2gray(img)
# img_binary = np.where(img_gray > 0.5, 1, 0)
# plt.subplot(221)
# plt.imshow(img_binary, cmap='gray')
# plt.show()





