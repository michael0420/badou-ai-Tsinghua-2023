# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:45:04 2023

@author: lhx
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# cv2.imread 参数 第一个参数 路径；第二个参数是一个标志，它指定了读取图像的方式。
print(cv2.IMREAD_COLOR)#加载彩色图像。任何图像的透明度都会被忽视。它是默认标志,可不写。1
print(cv2.IMREAD_GRAYSCALE)#以灰度模式加载图像 0
print(cv2.IMREAD_UNCHANGED)#加载图像，包括alpha通道 -1

src = "lenna.png"
img_cv = cv2.imread(src)
gray_img_cv = cv2.imread(src,0)
uc_img_cv = cv2.imread(src,-1)

print(img_cv.shape)
print(gray_img_cv.shape)
#print("b=",img_cv[:, :, 0])
#print("g=",img_cv[:, :, 1])
#print("r=",img_cv[:, :, 2])
#img_cv = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)
#print(img_cv.shape)
#print("r=",img_cv[:, :, 0])
#print("g=",img_cv[:, :, 1])
#print("b=",img_cv[:, :, 2])

h,w=img_cv.shape[0:2]#img_cv.shape[:2]
print("h=",h)
print("w=",w)

#grayImage = img_cv
#灰度图
grayImage = np.zeros(img_cv.shape, np.uint8)
#二值图
ezImage = np.zeros(img_cv.shape, np.uint8)
ezImage1 = gray_img_cv/255
for i in range(h):
    for j in range(w):
        #图像灰度化处理 手动
        # 浮点 R0.3+G0.59+B0.11
        # R G B
        gray = img_cv[i,j][0]*0.3+img_cv[i,j][1]*0.59+img_cv[i,j][2]*0.11
        grayImage[i,j] = int(gray)
        
        #三通道二值化
        if(gray/255<=0.5):
            ezImage[i,j] = 0
        else:
            ezImage[i,j] = 255
            
        #两通道二值化
        if(ezImage1[i,j]<=0.5):
            ezImage1[i,j] = 0
        else:
            ezImage1[i,j] = 1


#图像灰度化处理 插件
grayImage1 = cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)

#二值化  插件
# 全局阈值
# =============================================================================
# mask_all = cv2.threshold(src,             # 要二值化的图片
#                        thresh=127,           # 全局阈值
#                        maxval=255,           # 大于全局阈值后设定的值
#                        type=cv2.THRESH_BINARY)# 设定的二值化类型，
# =============================================================================
ret,mask_all = cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY)
#自适应阈值法
# =============================================================================
# mask_local=cv2.adaptiveThreshold(src=img,      # 要进行处理的图片
#                              maxValue=255, # 大于阈值后设定的值
# adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,# 自适应方法  
# thresholdType=cv2.THRESH_BINARY,             # 同全局阈值法中的参数一样
#                             blockSize=11,   # 方阵（区域）大小，
#                                    C=1)   # 常数项，
# =============================================================================
mask_auto=cv2.adaptiveThreshold(grayImage1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,1)  

#显示图像
#cv2.imshow("原图", img_cv)
# =============================================================================
# cv2.imshow("gray1", grayImage1)
# cv2.imshow("gray", grayImage)
# cv2.imshow("ez", ezImage)
# cv2.imshow("ez1", ezImage1)
# cv2.imshow("mask_all", mask_all)
# cv2.imshow("mask_auto", mask_auto)
# 
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================

#opencv图转为plt
grayImage = Image.fromarray(cv2.cvtColor(grayImage, cv2.COLOR_BGR2RGB))
grayImage1 = Image.fromarray(cv2.cvtColor(grayImage1, cv2.COLOR_BGR2RGB))
ezImage = Image.fromarray(cv2.cvtColor(ezImage, cv2.COLOR_BGR2RGB))

plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
# 灰度化
img_plt = plt.imread("lenna.png") 
img_gray = rgb2gray(img_plt)
plt.subplot(331),plt.title("plt灰度化")
plt.imshow(img_gray, cmap="gray")
plt.subplot(332),plt.title("手动灰度化")
plt.imshow(grayImage, cmap="gray")
plt.subplot(333),plt.title("cv2灰度化")
plt.imshow(grayImage1, cmap="gray")

plt.show()
# 二值化
rows, cols = img_gray.shape
for i in range(rows):
   for j in range(cols):
        if (img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
             img_gray[i, j] = 1

plt.subplot(334),plt.title("手动二值化")
plt.imshow(img_gray, cmap="gray")

img_binary = np.where(img_gray >= 0.5, 1, 0) 
plt.subplot(335),plt.title("plt二值化") 
plt.imshow(img_gray, cmap="gray")

plt.subplot(336),plt.title("cv2三通道二值化") 
plt.imshow(ezImage, cmap="gray")

plt.subplot(337),plt.title("cv2双通道二值化") 
plt.imshow(ezImage1, cmap="gray")

plt.subplot(338),plt.title("全局阈值（127）二值化") 
plt.imshow(mask_all, cmap="gray")

plt.subplot(339),plt.title("自适应阈值二值化") 
plt.imshow(mask_auto, cmap="gray")

plt.show()






