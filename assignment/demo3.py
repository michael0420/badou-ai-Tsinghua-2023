"""掉包实现RGB图片二值化"""
import numpy as np
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
img_gray = rgb2gray(img)#掉包直接实现RGB转gray
img_binary = np.where(img_gray >= 0.5, 1, 0)#掉包将gary转为二值
plt.imshow(img_binary, cmap='gray')#camp为颜色图谱
plt.show()