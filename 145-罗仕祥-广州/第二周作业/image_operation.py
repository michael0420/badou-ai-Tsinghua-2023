import cv2
import matplotlib.pyplot as plt
import numpy
from PIL import Image

# 调用接口实现RGB2GRAY、二值化
def image_from_api(image):

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('BGR')
    plt.axis('off')
    cv2.imwrite('BGR.png', image)

    # 将BGR格式的图像转换为RGB格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 2, 2)
    plt.imshow(rgb_image)
    plt.title('RGB')
    plt.axis('off')
    cv2.imwrite('RGB.png', rgb_image)

    # 将BGR格式的图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 2, 3)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Gray')
    plt.axis('off')
    cv2.imwrite('Gray.png', gray_image)

    # 二值化
    # 判断图像是否为灰度图像
    if len(image.shape) == 3:
        # OPENCV读取的图像是BGR格式的，所以需要将BGR格式的图像转换为灰度图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    plt.subplot(2, 2, 4)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary')
    plt.axis('off')
    cv2.imwrite('Binary.png', binary_image)

    plt.show()

# 从原理实现RGB2GRAY、二值化
def image_from_hand(image):
    # 读取图像
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('BGR')
    plt.axis('off')
    cv2.imwrite('BGR.png', image)

    rgb_image = image.copy()
    '''
    图像领域 img[:, :, :, i:]代表什么含义？
    img为BCHW格式的numpy 中的array类型，则上述[]里面，代表四个维度的切片范围，：表示整个维度都取，而最后一纬度 i: 代表取宽方向上取 第i列到最后1列。
    
    image / image[:] / image[:, :] 表示图像的原始数据。
    image[:, 0] / image[:, 1] / image[:, 2]：表示图像列的三通道像素。
    image[:, :, 0] / image[:, :, 1] / image[:, :, 2]：表示图像单个通道的像素。
    '''
    # 将BGR格式的图像转换为RGB格式，原理：将代表B和R的数字交换位置
    rgb_image[:, :, 0], rgb_image[:, :, 2] = image[:, :, 2], image[:, :, 0]
    plt.subplot(2, 2, 2)
    plt.imshow(rgb_image)
    plt.title('RGB')
    plt.axis('off')
    cv2.imwrite('RGB.png', rgb_image)

    # 将BGR格式的图像转换为灰度图像，使用公式：Y = 0.299 R + 0.587 G + 0.114 B
    h, w = image.shape[:2]
    gray_image = numpy.zeros((h, w), dtype=numpy.uint8)
    for i in range(h):
        for j in range(w):
            gray_image[i, j] = int(image[i, j][0] * 0.299 + image[i, j][1] * 0.587 + image[i, j][2] * 0.114)
    plt.subplot(2, 2, 3)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Gray')
    plt.axis('off')
    cv2.imwrite('Gray.png', gray_image)

    # 二值化
    h, w = image.shape[:2]
    binary_image = numpy.zeros((h, w), dtype=numpy.uint8)
    for i in range(h):
        for j in range(w):
            if gray_image[i, j] > 127:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0
    plt.subplot(2, 2, 4)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary')
    plt.axis('off')
    cv2.imwrite('Binary.png', binary_image)

    plt.show()

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('lenna.png')
    # 从接口实现RGB2GRAY、二值化
    image_from_api(image)
    # 从原理实现RGB2GRAY、二值化
    image_from_hand(image)