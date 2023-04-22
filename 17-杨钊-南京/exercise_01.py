#练习二(其中的rgb转灰度图以及灰度图转二值图不是手写算法）
from skimage.color import rgb2gray #从skimage.color导入rgb2gray
import numpy as np    #导入numpy作为np
import matplotlib.pyplot as plt#导入matplotlib.pyplot作为plt
from PIL import Image#从PIL导入Image
import cv2#导入opencv-python

#读取图像并且为灰度化做准备
image=cv2.imread("D:\image\lenna.png")#使用cv2.imread读入原图，以矩阵形式存入image
height,width=image.shape[:2]#使用shape函数获取读入图像的高度方向和宽度方向的像素数，分别存入height和width中
image_gray=np.zeros([height,width],image.dtype)#使用np.zero创建一个规格和原图相同的空白矩阵，赋值给对象image_gray
print(image_gray)#将image_gray以矩阵形式打印出来

#创建画布，并显示原图像
plt.subplot(221)#创建一个2*2画布，并在第一个子图上预计划显示
image=plt.imread("D:\image\lenna.png")#读入原图
plt.imshow(image)#预计划显示原图

#灰度化(调用函数）并且预显示
image_gray=rgb2gray(image)
plt.subplot(222)#创建一个2*2画布，并在第二个子图上预计划显示
plt.imshow(image_gray,cmap='gray')#预计划显示灰度图


#将灰度图二值化(调用函数）并且预显示
image_binary = np.where(image_gray >= 0.5, 1, 0)

#创建画布显示二值图
plt.subplot(223)
plt.imshow(image_binary,cmap='gray')

#显示整个画布
plt.show()



