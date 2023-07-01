#随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np
import cv2
from numpy import shape 
import random
def GaussianNoise(src,means,sigma,percetage):             #src：原图，percetage百分比（把图像中多少的点变成高斯噪声），means,sigma是参数，不要取太大
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])     #计算有多少个点需要做高斯噪声，src.shape[0]" 是用来获取图像的行数，即图像的高度，src.shape[1]" 是用来获取图像的列数，即图像的宽度
    for i in range(NoiseNum):
		#每次取一个随机点
		#把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
		#高斯噪声图片边缘不处理，故-1
        randX=random.randint(0,src.shape[0]-1)             #-1是因为序号是从0开始的，点的个数是512个
        randY=random.randint(0,src.shape[1]-1)
        #此处在原有像素灰度值上加上随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg
img = cv2.imread('lenna.png',0)     # 读取灰度图像
img1 = GaussianNoise(img,2,4,0.8)   #对图像添加高斯噪声
img = cv2.imread('lenna.png')       # 重新读取原始图像（彩色图像）
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 将彩色图像转换为灰度图像
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)           # 显示原始图像
cv2.imshow('lenna_GaussianNoise',img1)  # 显示添加高斯噪声后的图像
cv2.waitKey(0)                          #等待按键输入，当按下任意键时关闭窗口
