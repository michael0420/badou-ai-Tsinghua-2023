import cv2
import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("lenna.png")
# 原图
plt.subplot(221)
plt.imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_binary = np.where(img_gray >= 0.5, 1, 0)


# 灰度图
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

# 二值图
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')

plt.show()
