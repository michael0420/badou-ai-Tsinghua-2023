import cv2
import numpy as np


def fc(i, u, v):
    h, w = i.shape[:2]  # 只读取图像的高度和宽度信息
    emp = np.zeros((u, v, 3), np.uint8)
    sh = u / h
    sw = v / w
    for a in range(u):
        for b in range(v):
            x = int(a / sh + 0.5)
            y = int(b / sw + 0.5)
            emp[a, b] = i[x, y]
    return emp


im = cv2.imread("lenna.png")
zoom = fc(im, 1023, 1023)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", im)
cv2.waitKey(0)
