import cv2
import matplotlib.pyplot as plt
import numpy as np

# 自定义灰度化函数
def img2gray(filePath, gray_type):
    img = cv2.imread(filePath)
    h,w,c = img.shape
    img_gray1 = np.zeros([h,w],dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            c = img[i, j]
            if gray_type == "float":
                # 浮点算法 r*0.2 + g*0.36 + b*0.44
                img_gray1[i,j] = int(c[0]*0.2 + c[1]*0.36 + c[2]*0.44) #计算后向下取整
            elif gray_type == "avg":
                img_gray1[i, j] = (c[0] + c[1] + c[2])//3
            elif gray_type == "R":
                img_gray1[i, j] = c[2]
            elif gray_type == "G":
                img_gray1[i, j] = c[1]
            elif gray_type == "B":
                img_gray1[i, j] = c[0]
            elif gray_type == "max":
                img_gray1[i, j] = np.max(c)
            else:
                img_gray1[i, j] = int(c[0] * 0.3 + c[1] * 0.4 + c[2] * 0.3)
    return img_gray1

img = cv2.imread('dog1.webp')
# print(img.shape)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# print(img2.shape)

#自定义灰度化
img_gray_avg = img2gray('dog1.webp','avg')
img_gray_max = img2gray('dog1.webp','max')
img_gray_r = img2gray('dog1.webp','r')
img_gray_g = img2gray('dog1.webp','g')
img_gray_b = img2gray('dog1.webp','b')
img_gray_float = img2gray('dog1.webp','float')

#接口 灰度化实现
img_gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(img_gray2)

#二值化
img_binary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
min = np.min(img_binary)
max = np.max(img_binary)
h,w = img_binary.shape
for i in range(h):
    for j in range(w):
        if img_binary[i,j] > 128:
            img_binary[i,j] = max
        else:
            img_binary[i,j] = min


#作图
imgs = [img,img2,img_gray_avg,img_gray_max,img_gray_r,img_gray_g,img_gray_b,img_gray_float,img_gray2,img_binary]
titles = ["opencv read","after bgr2rgb","avg gray","max gray","r gray","g gray","b gray","float gray","opencv rgb2gray","after binary"]
w = len(imgs)//2
plt.figure(figsize=(2,w))
c = 0
for i in range(len(imgs)):
    c = c + 1
    plt.subplot(2,w,c)
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()





