from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
#显示原来图片
img = cv2.imread("C:\\Users\\xiaohao\\Desktop\\lenna.png")
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)

# 灰度化（调接口）
img_gray = rgb2gray(img)
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
# 灰度化（手动实现）
h,w = img.shape[:2]                               #获取图片的high和wide
img_gray = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j]                             #取出当前high和wide中的BGR坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
plt.subplot(221)
img = plt.imread("C:\\Users\\xiaohao\\Desktop\\lenna.png")
# img = cv2.imread("C:\\Users\\xiaohao\\Desktop\\lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)
# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
