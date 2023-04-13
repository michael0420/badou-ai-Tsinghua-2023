# _*_ coding: utf-8 _*_
# author: ding kou
# time: 2023-04-12
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.color import rgb2gray

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)


class ImageFactory:
    """ 图片工厂 """

    image_gray = None  # 灰度化结果，可以等于某个灰度化返回值

    def __init__(self, path: str = None):
        self.path = path
        self.plt_img = plt.imread(self.path)  # plt 读取图片
        self.cv2_img = cv2.imread(self.path)  # cv2 读取图片

    def init_image(self):
        """ 原始图片展示 """
        plt.subplot(231)
        # source_img = plt.imread(self.path)
        plt.imshow(self.plt_img)
        plt.title("原图样列展示")

    def base_analysis_gray_process(self):
        """ 底层原理解析灰度化过程 """
        # 灰度化基础写法
        img = self.cv2_img
        row, column = img.shape[:2]
        img_gray = np.zeros([row, column], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
        for i in range(row):
            for j in range(column):
                loc_bgr = img[i][j]  # 取出当前high和wide中的BGR坐标
                # todo 正常情况下灰度化图像颜色是 RGB -> (0.3, 0.59, 0.11)，而cv2读取后的是 BGR -> (0.11, 0.59, 0.3)
                # 将BGR坐标转化为gray坐标并赋值给新图像
                img_gray[i][j] = int(sum([loc_bgr[0] * 0.11, loc_bgr[1] * 0.59, loc_bgr[2] * 0.3]))
        print("image show gray: %s" % img_gray)
        plt.subplot(232)
        plt.imshow(img_gray, cmap='gray')
        plt.title("灰度化")
        return img_gray

    def image_gray_by_cv2(self):
        """ cv2 处理图片灰度化 """
        img = self.cv2_img
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cv2 调包使用
        plt.subplot(233)
        plt.imshow(img_gray, cmap='gray')
        plt.title("cv2灰度化")
        return img_gray

    def image_gray_by_skimage(self):
        """ bgr2gray 处理图片灰度化 """
        img_gray = rgb2gray(self.plt_img)
        plt.subplot(234)
        plt.imshow(img_gray, cmap='gray')
        plt.title("skimage灰度化")
        return img_gray

    def img_binary_analysis_code(self):
        """ 二值化 底层原理解析 """
        img_gray = rgb2gray(self.plt_img)
        rows, cols = img_gray.shape
        for i in range(rows):
            for j in range(cols):
                if (img_gray[i, j] <= 0.5):
                    img_gray[i, j] = 0
                else:
                    img_gray[i, j] = 1
        print(f'---二值化结果：')
        print(img_gray)

    def img_binary_by_gray(self):
        """ rgb2gray 二值化 """
        img_gray = rgb2gray(self.plt_img)
        img_binary = np.where(img_gray > 0.5, 1, 0)
        plt.subplot(235)
        plt.imshow(img_binary, cmap='gray')
        plt.title("skimage二值化展示")

    # def img_binary_by_cv2_gray(self):
    #     """ 自己构思了使用了均值 进行二值化 """
    #     img = self.cv2_img
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cv2 调包使用
    #     t1 = img_gray.flatten()  # 降维
    #     avg1 = t1.sum() / len(t1)
    #     # avg1 = 130
    #     img_cv2_binary = np.where(img_gray >= avg1, 1, 0)
    #     plt.subplot(236)
    #     plt.imshow(img_cv2_binary, cmap='gray')
    #     plt.title("cv2均值二值化化展示")

    def show(self):
        plt.show()


if __name__ == '__main__':
    # 这里将展示各个方式处理图片灰度化或者二值化，任意取即可
    imf = ImageFactory(path='./lenna.png')
    imf.init_image()  # 原始图片展示
    imf.base_analysis_gray_process()  # 底层原理解析灰度化过程展示
    imf.image_gray_by_cv2()  # cv2 处理图片灰度化
    imf.image_gray_by_skimage()  # bgr2gray 处理图片灰度化
    imf.img_binary_analysis_code()  # 二值化 底层原理解析
    imf.img_binary_by_gray()  # rgb2gray 二值化
    # imf.img_binary_by_cv2_gray()  # 自己构思了使用了均值 进行二值化，暂时不可取，还需调参数
    imf.show()
    pass

