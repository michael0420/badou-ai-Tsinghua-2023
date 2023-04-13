# 调用接口实现RGB2GRAY
# 调用OpenCV库的cv2.cvtColor()函数即可实现RGB2GRAY的操作。该函数具有以下参数：
# src：输入RGB彩色图像;
# code：颜色转换代码，本例中为cv2.COLOR_RGB2GRAY;
# dst：输出灰度图像;

import cv2
from PIL import Image


# 加载RGB图像
img = cv2.imread("pic/1.png")
# 将RGB图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 将numpy数组转换为Image对象，在转换时需指定模式"L"表示8位带符号整数的灰度图像
gray_img = Image.fromarray(gray_img.astype('uint8'), mode='L')
# 显示灰度图像
gray_img.show()
