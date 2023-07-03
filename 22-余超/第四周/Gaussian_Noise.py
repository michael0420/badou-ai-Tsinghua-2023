# 高斯噪声  概率密度函数符合高斯分布的一类
# 数字图像家上高斯噪声的步骤1.输入参数sigma、mean 2.生成高斯随机数 3.根据输入像素值计算出输出像素值 4. 重新将像素值放缩在[0,255] 5.循环所有像素 6.输出图像
import numpy as np
import cv2
from numpy import shape
import random
# 参数为：原图片（numpy格式）、高斯分布均值、高斯分布方差、信噪比
def Gaussian_Noise(src_img,mean,sigma,percentage):
    #先定义要输出的噪点图片     
    NoiseImg=src_img
    NoiseNum=int(percentage*src_img[0]*src_img[1])
    for i in range(NoiseNum):
        #取x Y坐标方向上NoiseNum个随机点
        randX=random.randint(0,NoiseImg.shape[0]-1)
        randY=random.randint(0,NoiseImg.shape[1]-1)
#         random.gauss()为输出高斯分布的一组数据，参数为高斯分布的均值与方差
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(mean,sigma)
        if NoiseImg[randX,randY]<0:
            NoiseImg[randX,randY]=0
        if NoiseImg[randX,randY]>255:
            NoiseImg[randX,randY]=255    
    return NoiseImg
