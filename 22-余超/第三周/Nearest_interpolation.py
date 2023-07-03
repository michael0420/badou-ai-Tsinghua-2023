import cv2
import numpy as np
#  最邻近插值,img：缩放前的原图，out_dim：缩放后的大小
def Nearest_Neighbor(img,out_dim):
    height,width,channels=img.shape
    if height==out_dim[0] and width==out_dim[1]:
        return img
    targetImg=np.zeros((out_dim[0],out_dim[1],channels),np.uint8)
# sh、sw 为插值算法的缩放比例   
    sh=out_dim[0]/height
    sw=out_dim[1]/width
    for i in range(channels):
        for j in range(out_dim[0]):
            for k in range(out_dim[1]):
# int(a+0.5)可以将浮点数a四舍五入            
                 x=int(i/sh+0.5)
                 y=int(j/sw+0.5)
                 targetImg[j,k,i]=img[x,y,i]
    return targetImg
