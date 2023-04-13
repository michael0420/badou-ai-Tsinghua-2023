import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

img = plt.imread("test.jpg")    # 原图
img_gray = rgb2gray(img)    # 灰度化
# print(img_gray)
img_binary = np.where(img_gray >= 0.5, 1, 0)    # 二值化
# print(img_binary)

plt.subplot(131), plt.imshow(img), plt.title("Original")
plt.subplot(132), plt.imshow(img_gray, "gray"), plt.title("Gray")
plt.subplot(133), plt.imshow(img_binary, "gray"), plt.title("Binary")
plt.show()
