'''
rgb2gray
'''

import cv2
import numpy as np


def showImage(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


def weight(img, blue_weight=0.11, green_weight=0.59, red_weight=0.3):
    if (blue_weight < 0 or blue_weight > 1) or (green_weight < 0 or green_weight > 1) or (red_weight < 0 or red_weight > 1):
        raise ValueError("权重错误，请重新输入")
    # 此处dtype为uint8，其他类型会显示异常，cv2.imshow中会对不同数据类型进行处理(可以直接使用img.dtype)
    target = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            target[i][j] = blue_weight * img[i][j][0] + green_weight * img[i][j][1] + red_weight * img[i][j][2]
    return target




if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    showImage(gray, 'gray')
    weightGray = weight(img)
    showImage(weightGray, 'weightGray')

    weightGreen = weight(img, 0, 1, 0)
    showImage(weightGreen, 'weightGreen')
