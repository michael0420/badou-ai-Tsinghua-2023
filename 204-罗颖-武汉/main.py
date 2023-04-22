import cv2
import matplotlib.pyplot as plt
import numpy as np


#灰度化
img = cv2.imread("2.bmp")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray,cmap='gray')
plt.show()

#二值化
ret,img_binary=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
plt.imshow(img_binary,cmap='gray')
plt.show()