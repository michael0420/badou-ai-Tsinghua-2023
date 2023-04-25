import cv2
import numpy as np

# 最邻近插值
def function(ima):
    # 获取图像的高度、宽度和通道数
    height, width, channels = ima.shape

    # 创建一个800x800像素大小的全零矩阵，作为空的图像存储空间
    emptyImage = np.zeros((800, 800, channels), np.uint8)

    # 计算高度和宽度缩放因子
    sh = 800 / height
    sw = 800 / width

    # 两个嵌套循环遍历800x800像素的空图像
    for i in range(800):
        for j in range(800):
            # 使用缩放因子将原始图像中的像素映射到新图像中
            x = int(i / sh)
            y = int(j / sw)
            emptyImage[i, j] = ima[x, y]

    # 返回缩放后的图像
    return emptyImage


# 读取名为“lenna.png”的图像文件
img = cv2.imread("lenna.png")

# 调用function函数缩放图像
zoom = function(img)

# 输出缩放后的图像矩阵和尺寸
print(zoom)
print(zoom.shape)

# 显示原始图像和缩放后的图像
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)

# 等待按键关闭窗口
cv2.waitKey(0)