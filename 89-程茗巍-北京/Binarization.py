import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def Binarization(file,threshold):
    #打开图像
    im = Image.open(file)

    #获取图像高度、宽度和数据
    width, height = im.size
    mode = im.mode
    data = list(im.getdata())
    Gray = []

    for i in data:
        gray_value = float(i[0]*30 + i[1]*59 + i[2]*11) / 100
        Gray.append(gray_value)

    #将处理后的list转换成像素矩阵
    gray_image = np.array(Gray).reshape(width,height)

    binarization = np.zeros(gray_image.shape)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i][j] <= threshold:
                binarization[i][j] = 0
            else:
                binarization[i][j] = 255
    return binarization



if __name__ =='__main__':
    filename = 'lenna.png'
    threshold = 128
    im = Image.open(filename)

    #二值化
    binarization_im = Binarization(filename,threshold)
    # 显示原始图像和处理后的图像
    fig, ax = plt.subplots(1, 2)
    print(ax)
    ax[0].imshow(im)
    ax[1].imshow(binarization_im, cmap='gray')
    plt.show()

    #接口实现
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # 对灰度图像进行二值化
    ret, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    # 显示原始图像和二值化后的图像
    cv2.imshow("Binary Image", img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()