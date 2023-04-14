# RGB2GRAY是将RGB三通道彩色图像转换为灰度图像的过程
# 其中RGB三通道中每个像素值的范围是[0，255]
# 而灰度图像每个像素值的范围是[0，255]


import numpy as np
from PIL import Image


def rgb2gray(img):
    # 转换成灰度图像的权值系数，分别对应三个通道（R、G、B）
    coefficients = [0.299, 0.587, 0.114]
    img_arr = np.array(img)
    gray_arr = np.dot(img_arr[..., :3], coefficients)
    # 灰度图像要将二维数组转换成Image对象，在转换时需指定模式"L"表示8位带符号整数的灰度图像
    gray_img = Image.fromarray(gray_arr.astype('uint8'), mode='L')

    return gray_img

# 加载RGB图像
img = Image.open("pic/1.png")
# 将RGB图像转换为灰度图像
gray_img = rgb2gray(img)
# 显示灰度图像
gray_img.show()
