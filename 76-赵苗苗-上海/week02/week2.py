#彩色图像的灰度化、二值化
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
#灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]                               #获取图片的high和wide，img.shape[:2]是切片操作，保留前两个元素
img_gray = np.zeros([h,w],img.dtype)              #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j]                               #取出当前high和wide中的BGR坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)

plt.subplot(221)                                  #plt.subplot(221)​是用于创建一个具有2行2列的子图网格，并选择其中的第1个子图来进行绘制
img = plt.imread("lenna.png") 
# img = cv2.imread("lenna.png", False) 
plt.imshow(img)
print("---image lenna----")
print(img)


#二值化
rows, cols = img_gray.shape     #获取灰度图像的行数和列数，并将其分别赋值给变量 ​rows​和 ​cols​
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)     #实现了将灰度图像二值化的功能，将像素值大于等于0.5的部分设为1，小于0.5的部分设为0
print("-----image_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223) 
plt.imshow(img_binary, cmap='gray')    #​​plt.imshow()​函数用于显示图像数据。它接受一个二维或三维的数组作为输入，并将其显示为图像。如果输入是二维数组，则被视为灰度图像；如果输入是三维数组，则被视为彩色图像。可以通过设置参数来调整图像的颜色映射、插值方法等。
plt.show()                             #​​plt.show()​函数用于显示所有已创建的图像。它会打开一个图形窗口，并将所有的图像显示在该窗口中。在调用 ​plt.show()​之前，可以使用 ​plt.plot()​、​plt.imshow()​等函数创建多个图像。一旦调用 ​plt.show()​，所有已创建的图像将一起显示出来。
