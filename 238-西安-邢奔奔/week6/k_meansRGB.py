#! /user/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/aragaki/artificial/image/lenna.png')
print(img.shape)


data = img.reshape(-1,3)
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

flags = cv2.KMEANS_RANDOM_CENTERS

compactness2, label2, centers2 = cv2.kmeans(data,2,None,criteria,10,flags)

compactness4, label4, centers4 = cv2.kmeans(data,4,None,criteria,10,flags)

compactness8, label8, centers8 = cv2.kmeans(data,8,None,criteria,10,flags)

compactness16, label16, centers16 = cv2.kmeans(data,16,None,criteria,10,flags)

compactness64, label64, centers64 = cv2.kmeans(data,64,None,criteria,10,flags)

#   图像转换回uint8二维类型,通过flatten做扁平化
centers2 = np.uint8(centers2)
res = centers2[label2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
label4con = label4.flatten()
res = centers4[label4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[label8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[label16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[label64.flatten()]
dst64 = res.reshape((img.shape))
#   如果是cv2.imshow则不需要转换，plt.imshow需要转换为RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)
#   用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#   显示图像
titles = ['原始图像', '聚类图像 K=2', '聚类图像 K=4',
          '聚类图像 K=8', '聚类图像 K=16',  '聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
   plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()