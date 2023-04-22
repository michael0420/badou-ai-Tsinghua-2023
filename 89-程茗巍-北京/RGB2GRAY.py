from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def RGB2GRAY(file):
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
    gray_im = np.array(Gray).reshape(width,height)


    return gray_im


if __name__ =='__main__':
    filename = 'lenna.png'
    im = Image.open(filename)
    #RGB2GRAY
    gray_im = RGB2GRAY(filename)

    # 显示原始图像和处理后的图像
    fig, ax = plt.subplots(1, 2)
    print(ax)
    ax[0].imshow(im)
    ax[1].imshow(gray_im, cmap='gray')
    plt.show()

    #cv 接口实现
    img_color = cv2.imread("lenna.png")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()