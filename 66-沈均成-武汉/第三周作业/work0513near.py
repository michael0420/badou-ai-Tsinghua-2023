import cv2
import numpy as np


def nearest(img):
    h, w, c = img.shape
    empty_img = np.zeros((800, 800, c), np.uint8)
    sh = h/800
    sw = w/800
    # print(sh,sw)
    for i in range(800):                      #利用插值后（800*800）的坐标计算插值前（h*w）的坐标
        for j in range(800):
            x = int(i*sh + 0.5)               #i后/i前 = 800/h,j后/j前 = 800/w
            y = int(j*sw + 0.5)
            empty_img[i, j] = img[x, y]
    return empty_img


img = cv2.imread("lenna.png")
zoom = nearest(img)
cv2.imshow("nearest", zoom)
cv2.imshow("img", img)
cv2.waitKey(0)
