import cv2
import numpy as np
import math

def canny_detail(img):

    #1.高斯滤波
    sigma = 0.5
    dim = int(np.round(6 * sigma) + 1)
    if dim % 2 == 0:
        dim += 1
    gauss_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            gauss_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    gauss_filter = gauss_filter / gauss_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros([dx, dy])
    img_padding = np.pad(img, ((dim // 2, dim // 2), (dim // 2, dim // 2)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_padding[i:i+dim, j:j+dim] * gauss_filter)
    cv2.imshow("gauss", img_new)

    #2.水平、垂直和对角边缘
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_gradientX = np.zeros(img_new.shape)
    img_gradientY = np.zeros(img_new.shape)
    img_gradient = np.zeros(img_new.shape)
    img_padding = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_gradientX[i, j] = np.sum(img_padding[i: i + 3, j: j + 3] * sobel_x)
            img_gradientY[i, j] = np.sum(img_padding[i: i + 3, j: j + 3] * sobel_y)
            img_gradient[i, j] = np.sqrt(img_gradientY[i, j] ** 2 + img_gradientX[i, j] ** 2)
    img_gradientX[img_gradientX == 0] = 0.00000001
    angle = img_gradientY / img_gradientX
    cv2.imshow("gradient", img_gradient)

    #3.非极大值抑制
    img_nms = np.zeros(img_gradient.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_gradient[i-1:i+2, j-1:j+2]
            if angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not(img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not(img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not(img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            if flag:
                img_nms[i, j] = img_gradient[i, j]
    cv2.imshow("nms", img_nms)

    #4.用双阈值检测和连接边缘
    low_boundary = img_nms.mean() * 0.5
    high_boundary = low_boundary * 3
    stack = []
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            if img_nms[i, j] >= high_boundary:
                img_nms[i, j] = 255
                stack.append([i, j])
            elif img_nms[i, j] <= low_boundary:
                img_nms[i, j] = 0
    while not len(stack) == 0:
        temp1, temp2 = stack.pop()
        a = img_nms[temp1 - 1: temp1 + 2, temp2 - 1: temp2 + 2]
        if a[0, 0] > low_boundary and a[0, 0] < high_boundary:
            img_nms[temp1 - 1, temp2 - 1] = 255
            stack.append([temp1 - 1, temp2 - 1])
        if a[0, 1] > low_boundary and a[0, 1] < high_boundary:
            img_nms[temp1 - 1, temp2] = 255
            stack.append([temp1 - 1, temp2])
        if a[0, 2] > low_boundary and a[0, 2] < high_boundary:
            img_nms[temp1 - 1, temp2 + 1] = 255
            stack.append([temp1 - 1, temp2 + 1])
        if a[1, 0] > low_boundary and a[1, 0] < high_boundary:
            img_nms[temp1, temp2 - 1] = 255
            stack.append([temp1, temp2 - 1])
        if a[1, 2] > low_boundary and a[1, 2] < high_boundary:
            img_nms[temp1, temp2 + 1] = 255
            stack.append([temp1, temp2 + 1])
        if a[2, 0] > low_boundary and a[2, 0] < high_boundary:
            img_nms[temp1 + 1, temp2 - 1] = 255
            stack.append([temp1 + 1, temp2 - 1])
        if a[2, 1] > low_boundary and a[2, 1] < high_boundary:
            img_nms[temp1 + 1, temp2] = 255
            stack.append([temp1 + 1, temp2])
        if a[2, 2] > low_boundary and a[2, 2] < high_boundary:
            img_nms[temp1 + 1, temp2 + 1] = 255
            stack.append([temp1 + 1, temp2 + 1])
    for i in range(dx):
        for j in range(dy):
            if img_nms[i, j] != 0 and img_nms[i, j] != 255:
                img_nms[i, j] = 0
    return img_nms


if __name__ == '__main__':
    img = cv2.imread("lenna.png",0)
    img_1 = cv2.Canny(img, 80, 80)

    #img = img.mean(axis=-1)
    img_2 = canny_detail(img)

    cv2.imshow("source", img)
    cv2.imshow("canny1", img_1)
    cv2.imshow("canny2", img_2)
    cv2.waitKey(0)
