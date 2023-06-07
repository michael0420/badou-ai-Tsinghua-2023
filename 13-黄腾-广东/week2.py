"""
作者
@author： YelloWiwi
实现彩色图像的灰度化、二值化
RGB2GRAY、Binarization
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2


"""
演示不调接口的人工灰度化
"""
# 使用像素模式打开lenna图片文件
img = cv2.imread("lenna.png")
# 赋值高、宽的像素长度
# shape的用法------输出长、宽、通道数--(rows, columns, channels)
h, w = img.shape[:2]  # high, wide
# 创建一个与原图像一致的单通道为0的矩阵
# np.zeros的用法------输入形状、数据类型对象、优先列或行排序--np.zeros(shape, dtype, order)
# img.dtype指img的数据类型对象
img_gray = (np.zeros((h, w), img.dtype))
for i in range(h):
    for j in range(w):
        # 取出每一个像素点的坐标
        lo = img[i, j]  # location
        # 将每一个坐标的B G R 赋值给空白的灰度值图片
        img_gray[i, j] = int(lo[0]*0.11 + lo[1]*0.59 + lo[2]*0.3)
print(img_gray)
# 对于cv2.imshow的声明:第一个参数是一个窗口名称（也就是我们对话框的名称），它是一个字符串类型。第二个参数是我们的图像。您可以创建任意数量的窗口，但必须使用不同的窗口名称。
cv2.imshow("image show gray", img_gray)

# waitKey()的作用声明：
# 1、waitKey()–这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下 键,则接续等待(循环）
# 2、如果设置waitKey(0),则表示程序会无限制的等待用户的按键事件
# cv2.waitKey()


# 221: 分成二行二列，位于第一个部分
plt.subplot(221)
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化调接口
plt.subplot(222)
img_gray = rgb2gray(img)
plt.imshow(img_gray, cmap='gray')

# 二值化手动赋值原理
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1


# 二值化调接口
img_binary = np.where(img_gray <= 0.5, 0, 1)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')


# 展示221，222，223
plt.show()
