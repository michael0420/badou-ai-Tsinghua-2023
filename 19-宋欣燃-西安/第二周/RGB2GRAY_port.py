"""
@author:songxinran

彩色图像灰度化-接口实现
"""
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img = plt.imread("lenna.png") # cv.imread("lenna.png")是按照BGR方式读取的，颜色和原图不同
plt.subplot(221)
plt.imshow(img)

# img_gray = rgb2gray(img) # 接口一
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 接口二

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.show()







