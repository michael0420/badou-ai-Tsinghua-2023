"""手写实现RGB图片二值化"""
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("lenna.png")#读
img_gray = rgb2gray(img)#掉包直接实现RGB转gray
h,w = img.shape[:2]#读高宽
img_binary = np.zeros([h,w],img.dtype)#创建一个数组用于存放灰度图转换为二值图
for i in range(h):#二值化
      for j in range(w):
          if (img_gray[i, j] <= 0.5):
              img_binary[i, j] = 0
          else:
              img_binary[i, j] = 1
plt.imshow(img_binary, cmap='gray')#camp为颜色图谱
plt.show()
