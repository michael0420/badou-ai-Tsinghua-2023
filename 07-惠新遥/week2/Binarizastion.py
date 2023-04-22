
"""
彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# img = cv2.imread("lenna.png")
img = cv2.imread("LowResolution.png")
# img = cv2.imread("SelfProducedStarChat.png")

# 灰度化
img_gray = rgb2gray(img)

# 二值化
img_binary = np.where(img_gray >= 0.8, 1, 0)

# 展示二值化结果图片
# plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()




