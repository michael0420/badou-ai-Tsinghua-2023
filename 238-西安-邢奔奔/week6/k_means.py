# !/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/aragaki/artificial/image/lenna.png',0)
print(img.shape)

rows,cols = img.shape[:]

# 将数据转为一维
data  = img.reshape(rows*cols,1)
data = np.float32(data)


criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#初始中心点随机选择
flags = cv2.KMEANS_RANDOM_CENTERS

'''
retval, labels, centers = cv2.kmeans(data, K, criteria, attempts, flags)
            kmeans参数和返回值：
            retval：用于评估kmeans效果
            labels： 输出标签矩阵，取值范围在0，K-1之间
            centers： 中心值数组
            data：  待聚类数据
            K： 要分的簇数
            criteria：终止条件  定义迭代终止的条件，通过元组(type, max_iter, epsilon)表示。type指定终止的类型，
            可以是cv2.TERM_CRITERIA_EPS（达到精确度要求）或cv2.TERM_CRITERIA_MAX_ITER（达到最大迭代次数）。max_iter是最大迭代次数，epsilon是所需的精度
            attempts：重复算法的次数，每次使用不同的初始中心点。最终结果将基于最低误差的一次迭代
            flags：初始中心点选择的标志，可以是cv2.KMEANS_RANDOM_CENTERS（随机选择初始化中心）或cv2.KMEANS_PP_CENTERS（使用k-means++算法选择初始化中心）

'''
compactness, labels, centers = cv2.kmeans(data,4,None,criteria,10,flags)

dst = labels.reshape(img.shape[0],img.shape[1])
#   这里中文显示乱码，添加字库后正常
plt.rcParams['font.sans-serif'] = ['SimHei']


#   显示图像
titles = ['原始图像','聚类图像']

imgs = [img,dst]
for i in range(2):
    plt.subplot(2,3,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    #隐藏坐标轴
img_con = np.zeros((512,512))
for i in range(256,512):
    for j in range(512):
        img_con[i][j] = 125
plt.subplot(2,3,3)
plt.imshow(img_con,'gray')
# img_con = img_con + 1
# for i in range(512):
#      for j in range(255):
#          img_con[i][j] = 3
# plt.subplot(2,2,3)
# plt.imshow(img_con,'gray')
# img = np.zeros((256,256))
# for i in range(256):
#     for j in range(256):
#         img[i][j] = (i+j)//2
# plt.subplot(2,2,4)
# plt.imshow(img,'gray')
plt.show()