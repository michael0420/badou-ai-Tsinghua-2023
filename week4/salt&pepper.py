
import numpy as np
import cv2
from numpy import shape
import random

def fun1(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])   #定义fun1函数，向图像中添加椒盐噪声
    
    for i in range(NoiseNum):                                 #创建变量NoiseNum，用于存储加噪声后的图像
        # 每次取一个随机点 
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 椒盐噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)              #使用循环遍历图像的随机像素点，将其随机设置为黑色或白色，模拟椒盐噪声。
        
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    
    return NoiseImg

img = cv2.imread('lenna.png', 0)                  #返回添加噪声后的图像
img1 = fun1(img, 0.2)                             #调用fun1函数，将原始图像和噪声百分比作为参数传递给该函数，生成添加椒盐噪声后的图像img1

# 在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
cv2.imwrite('lenna_PepperandSalt.png', img1)

img = cv2.imread('lenna.png')            #读取名为lenna.png的彩色图像
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #将彩色图像转换为灰度图像img2
cv2.imshow('source', img2)                     #使用opencv的inshow函数显示原始图像
cv2.imshow('lenna_PepperandSalt', img1)        #使用opencv的inshow函数显示添加椒盐噪声后的图像
cv2.waitKey(0)                                 #等待按键输入，当按下任意键时关闭窗口 