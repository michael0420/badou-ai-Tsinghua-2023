# 第四周作业

#1、 临近插值算法
#2、 双线性插值算法
#3、 几何中心对称系数证明
#4、 直方图均衡化

import cv2
import numpy as np

# 1、 临近插值算法
def nearest_interpolation(img, dstH, dstW):
    scrH, scrW, _ = img.shape
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = round(i * (scrH / dstH))
            scry = round(j * (scrW / dstW))
            retimg[i, j] = img[scrx, scry]
    return retimg

# 2、 双线性插值算法
def bilinear_interpolation(img, dstH, dstW):
    scrH, scrW, _ = img.shape
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = i * (scrH / dstH)
            scry = j * (scrW / dstW)
            x = int(scrx)
            y = int(scry)
            u = scrx - x
            v = scry - y
            retimg[i, j] = (1 - u) * (1 - v) * img[x, y] + u * (1 - v) * img[x + 1, y] + (1 - u) * v * img[x, y + 1] + u * v * img[x + 1, y + 1]
    return retimg

# 4、 直方图均衡化
def hist_equal(img):
    scrH, scrW, _ = img.shape
    retimg = np.zeros((scrH, scrW, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    
    return np.hstack([gray, dst])

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dstH = 700
    dstW = 700
    # nearest_interpolation
    ret = nearest_interpolation(img, dstH, dstW)
    cv2.imshow("nearest_interpolation", ret)
    # bilinear_interpolation
    ret = bilinear_interpolation(img, dstH, dstW)
    cv2.imshow("bilinear_interpolation", ret)
    # hist_equal
    ret = hist_equal(img)
    cv2.imshow("hist_equal", ret)
    cv2.waitKey(0)
    cv2.destroyAllWindows()