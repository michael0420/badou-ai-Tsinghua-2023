import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)  #生成一个800 x 800的全零矩阵
    sh=800/height
    sw=800/width                                      #对高和宽进行比例缩放
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5)
            y=int(j/sw + 0.5)                        #遍历得到function为800x800的图像上的点，这里加上0.5是防止int向下取整
            emptyImage[i,j]=img[x,y]                 #[i,j]是function为800x800的图像上的点，这里是看下对应源图像上的点在哪里，最临近插值得到的输出图像中的点的像素值都来自输入图像中最临近的点的像素值
    return emptyImage
img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)


