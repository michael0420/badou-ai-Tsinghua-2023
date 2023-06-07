import cv2
import numpy
from skimage.color import rgb2gray

# 读取文件
img = cv2.imread('img/lenna.png')
# 获取图像矩阵的行和列 通道数本次作业不用 截取掉 图像为三通道图
h,w = img.shape[:2]
# 通过numpy 创建一个 与 目标图像 相同数据类型 相同行列的全0矩阵
img_gray = numpy.zeros([h, w], img.dtype)
img_gray_normal = numpy.zeros([h, w], img.dtype)
img_binaryzation = numpy.zeros([h, w], img.dtype)

# 接口调用实现 灰度化
img_gray_2 = rgb2gray(img)
img_gray_3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in range(h):
    for j in range(w):
        # 解构 三通道图
        b, g, r = img[i][j]
        # 灰度化
        img_gray[i][j] = int(0.3 * r + 0.59 * g + 0.11 * b)

# 通过numpy实现归一化 data_normalized = (data - data_min) / (data_max - data_min)
img_gray_max, img_gray_min = numpy.max(img_gray), numpy.min(img_gray)
img_gray_normal = (img_gray - img_gray_min) / (img_gray_max - img_gray_min)

# 二值化
for i in range(h):
    for j in range(w):
        img_binaryzation[i][j] = 0 if img_gray_normal[i][j] <= 0.5 else 1

# 接口调用实现二值化
img_binaryzation_1 = numpy.where(img_gray_2 <= 0.5, 0, 1)

# 结果测试
print(img, '<-------img')
print(img_gray, '<-----img_gray')
print(img_gray_normal, '<----img_gray_normal')
print(img_gray_2, '<----img_gray_2')
print(img_gray_3, '<----img_gray_3')
print(img_binaryzation, '<----img_binaryzation')
print(img_binaryzation_1, '<-----img_binaryzation_1')
cv2.imshow('测试 渲染灰度图', img_gray)
# cv2.imshow('img_binary', img_binaryzation)
cv2.waitKey()