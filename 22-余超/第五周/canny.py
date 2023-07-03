# canny 边缘检测算法
# 1.图片灰度化 2.高斯滤波 3.调用sobel算子检测图片中水平、垂直、对角边缘  4.对梯度幅度进行非极大值抑制  5.双阈值算法检测、连接边缘
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
# 灰度化
def graying(src):
    w,h=src.shape[:2]
    img_gray=np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            m=src[i,j]
#           图片读取方式为rgb  
            img_gray[i,j]=m[0]*0.11+m[1]*0.59+m[2]*0.3
    return img_gray

# 高斯滤波 参数：src 需要处理的灰度化图像 sigma 高斯函数标准差 length:卷积核大小 返回值：高斯平滑后的图片
def Gaussian_smoothing(src,sigma,length):
#   k代表
    k=length//2
    w,h=src.shape#图片长宽
    guassian = np.zeros((length,length)) #定义高斯卷积核
    for i in range(length): #计算卷积核
        for j in range(length):
            guassian[i,j] = math.exp((-1)*((i-k)**2+(j-k)**2/2*sigma**2))
    guassian/=(2*math.pi*sigma**2)
    guassian=guassian/np.sum(guassian)                               
    new_img=np.zeros((w-k*2,h-k*2))#新图片由于没有padding所以图片大小以要比原来小2k
    for i in range(w-k*2):
        for j in range(h-k*2):
            new_img[i,j]=np.sum(src[i:i+length,j:j+length]*guassian)                   
    new_img=np.uint8(new_img)    #在进行np.sum卷积操作之后可能会得到小数，np.uint8将图片矩阵所有值变为0-255的整数    
    return new_img
  # 求梯度  参数：   返回值：梯度、方向  参数:灰度化图片  返回：梯度矩阵与方向矩阵
#    -1  0  1       1  2  1 
# Sx=-2  0  2    Sy=0  0  0 
#    -1  0  1      -1 -2 -1
def gradient(img):
    Sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Sobel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    w,h=img.shape
    img_gradient=np.zeros((w-2,h-2)) #此处可以优化，加一圈padding
    img_direction=np.zeros((w-2,h-2))
    for i in range(w-2):         #相当于进行了一次卷积操作
        for j in range(h-2):
            temp=img[i:i+3,j:j+3]  # 与Sobel算子计算的3*3图片矩阵
            dx=np.sum(np.multiply(temp,Sobel_x))
            dy=np.sum(np.multiply(temp,Sobel_y))
            img_gradient[i,j]=np.sqrt(dx**2+dy**2)#np.sqrt函数为算数平方根
            if dx!=0:img_direction[i,j]=dy/dx#这里的方向函数用tanζ来表示
            else:img_direction[i,j]=1000000
    img_gradient = np.uint8(img_gradient)            
    return  img_gradient,img_direction
  # NMS 非极大值抑制  通过遍历梯度矩阵中每一点，并放到八点邻域中与梯度方向上其余点
# （梯度方向与领域其余点横纵连线交点）进行比较，结果极大值保存、非极大值置为0
# 参数：gradient,direction为梯度矩阵与方向矩阵   返回：在8邻域中对比为极大值的梯度保存，其余为0
def NMS(gradient,direction):
    w,h=gradient.shape
    nms=np.zeros((w,h)) #在后续操作中nms矩阵最边缘一圈不会被操作，在后续循环中操作的nms[i,j]中的i、j取不到0和w、h
    for i in range(1,w-1):
        for j in range(1,h-1):
            flag=True
            temp=gradient[i-1:i+2,j-1:j+2] #8领域 3*3矩阵
#             在领域中用插值算法求在[i,j]梯度线与领域相交的两个点的梯度值，并与gradient[i,j]进行比较
#             g1、g2、g3、g4四点坐标依据梯度线方向不同而不同
            if direction[i,j]>=1:
                dtemp1=int(temp[0,1])+(int(temp[0,2])-int(temp[0,1]))/int(direction[i,j])
                dtemp2=int(temp[2,1])+(int(temp[2,0])-int(temp[2,1]))/int(direction[i,j])
                if not(temp[1,1]>=dtemp1 and temp[1,1]>=dtemp2):
                    flag=False
            elif direction[i,j]>=0:
                dtemp1=int(temp[1,2])+(int(temp[0,2])-int(temp[1,2]))*int(direction[i,j])  
                dtemp2=int(temp[1,0])+(int(temp[2,0])-int(temp[1,0]))*int(direction[i,j])
                if not(temp[1,1]>=dtemp1 and temp[1,1]>=dtemp2):
                    flag=False
            elif direction[i,j]>=-1:
                dtemp1=int(temp[1,2])+(int(temp[2,2])-int(temp[1,2]))*int(direction[i,j])  
                dtemp2=int(temp[1,0])+(int(temp[0,0])-int(temp[1,0]))*int(direction[i,j])
                if not(temp[1,1]>=dtemp1 and temp[1,1]>=dtemp2):
                    flag=False 
            else:
                dtemp1=int(temp[0,1])+(int(temp[0,0])-int(temp[0,1]))/int(direction[i,j])  
                dtemp2=int(temp[2,1])+(int(temp[2,2])-int(temp[2,1]))/int(direction[i,j])
                if not(temp[1,1]>=dtemp1 and temp[1,1]>=dtemp2):
                    flag=False
            if flag:
                nms[i,j]=gradient[i,j]       
            
    return nms
  # 双阈值检测 在nms非极大值抑制之后 通过设置的两个阈值：threshold1、threshold2 将nms矩阵数值中划分三类点
# 强边缘点：梯度值大于threshold2,该点大概率是边缘，设置为255
# 弱边缘点：梯度值介于threshold1和threshold2之间，有可能是真边缘（附近有边缘点）有可能是假边缘（孤立存在）
# 非边缘点：梯度值低于threshold1，置为0
def double_threshold(nms,threshold1,threshold2):
    w,h=nms.shape
    dst_img=np.zeros((w,h))
    zhan=[] #自定义栈，先存放强边缘点、后放弱边缘点，确保入栈的都是满足双阈值的点
    for i in range(w):
        for j in range(h):
            if nms[i,j]>=threshold2:
                dst_img[i,j]=255
                zhan.append([i,j])
            elif nms[i,j]<threshold1:
                dst_img[i,j]=0
    
    while len(zhan)!=0:
        temp_x,temp_y=zhan.pop()
        a=nms[temp_x-1:temp_x+2,temp_y-1:temp_y+2]
        if(a[0,0]<threshold2 and a[0,0]>=threshold1): #关于这里为什么要加条件a[0,0]<threshold2—不加的话强边缘点会一直入栈导致栈溢出
            dst_img[temp_x-1,temp_y-1]=255
            zhan.append([temp_x-1,temp_y-1])
        if(a[0,1]<threshold2 and a[0,1]>=threshold1): 
            dst_img[temp_x-1,temp_y]=255
            zhan.append([temp_x-1,temp_y])    
        if(a[0,2]<threshold2 and a[0,2]>=threshold1): 
            dst_img[temp_x-1,temp_y+1]=255
            zhan.append([temp_x-1,temp_y+1])   
        if(a[1,0]<threshold2 and a[1,0]>=threshold1): 
            dst_img[temp_x,temp_y-1]=255
            zhan.append([temp_x,temp_y-1])   
        if(a[1,2]<threshold2 and a[1,2]>=threshold1): 
            dst_img[temp_x,temp_y+1]=255
            zhan.append([temp_x,temp_y+1])       
        if(a[2,0]<threshold2 and a[2,0]>=threshold1): 
            dst_img[temp_x+1,temp_y-1]=255 
            zhan.append([temp_x+1,temp_y-1])   
        if(a[2,1]<threshold2 and a[2,1]>=threshold1): 
            dst_img[temp_x+1,temp_y]=255
            zhan.append([temp_x+1,temp_y])   
        if(a[2,2]<threshold2 and a[2,2]>=threshold1): 
            dst_img[tem_x+1,temp_y+1]=255
            zhan.append([temp_x+1,temp_y+1])
            
    return dst_img   
  from PIL import Image

if __name__ =="__main__":
    img=cv.imread("lenna.jpg",0)
    cv.imshow("original",img)
    smoothed_image=Gaussian_smoothing(img,0.5,3)
    cv.imshow("GuassinSmooth",smoothed_image)
    gradients ,direction = gradient(smoothed_image)
  # 双阈值的确定：
    threshold1=gradients.mean()*0.5
    threshold2=threshold1*3
    dst_img=double_threshold(NMS(gradients,direction),threshold1,threshold2)
    cv.imshow("output",dst_img)
