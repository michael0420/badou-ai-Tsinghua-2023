import cv2

# 读取灰度图像
image = cv2.imread("pic/1.png", 0)

# # 阈值处理二值化
thresh, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 显示二值化图像
cv2.imshow('二值化结果', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#
# src：输入灰度图像；
# thresh：二值化阈值；
# maxval：像素值大于阈值时要赋予的新值；
# type：二值化类型，本例中为cv2.THRESH_BINARY，表示大于阈值的像素赋值为maxval，小于阈值的像素赋值为0；

# import cv2
# import matplotlib.pyplot as plt
#
# # 读取灰度图像
# image = cv2.imread("pic/1.png", 0)
#
# # 绘制像素值分布直方图
# plt.hist(image.ravel(), 256, [0,256])
# plt.show()
#
# # 阈值处理二值化
# thresh, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
#
# # 绘制像素值分布直方图
# plt.hist(binary_image.ravel(), 256, [0,256])
# plt.show()
#
# # 显示二值化图像
# cv2.imshow('Binary Image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
