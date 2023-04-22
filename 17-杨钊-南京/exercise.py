#练习一(其中的rgb转灰度图以及灰度图转二值图是手写算法）
#导入所需的库
from skimage.color import rgb2gray #从skimage.color导入rgb2gray
import numpy as np    #导入numpy作为np
import matplotlib.pyplot as plt#导入matplotlib.pyplot作为plt
from PIL import Image#从PIL导入Image
import cv2#导入opencv-python


#读取图像并灰度化
image=cv2.imread("D:\image\lenna.png")#使用cv2.imread读入原图，以矩阵形式存入image
height,width=image.shape[:2]#使用shape函数获取读入图像的高度方向和宽度方向的像素数，分别存入height和width中
image_gray=np.zeros([height,width],image.dtype)#使用np.zero创建一个规格和原图相同的空白矩阵，赋值给对象image_gray
for i in range(height):#使用for循环将原图中的每个像素的b,g,r数值转化为gray值,填入image_gray的每个像素值中
    for j in range(width):
        m=image[i,j]
        image_gray[i,j]=int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(image_gray)#将image_gray以矩阵形式打印出来



#创建画布，并显示原图像
plt.subplot(221)#创建一个2*2画布，并在第一个子图上预计划显示
image=plt.imread("D:\image\lenna.png")#读入原图
plt.imshow(image)#预计划显示原图


#创建画布，并显示灰度图
plt.subplot(222)#创建一个2*2画布，并在第二个子图上预计划显示
plt.imshow(image_gray,cmap='gray')#预计划显示灰度图


#对灰度图进行二值化处理
height_gray,width_gray=image_gray.shape[:2]
for i in range(height_gray):
    for j in range(width_gray):
        if(image_gray[i,j]>=130.5):
            image_gray[i,j]=1
        else:
            image_gray[i,j]=0
image_binary=np.zeros([height,width],image.dtype)
for i in range(height):
    for j in range(width):
        image_binary[i,j]=image_gray[i,j]
print(image_binary)


#创建画布显示二值图
plt.subplot(223)
plt.imshow(image_binary,cmap='gray')

#显示整个画布
plt.show()