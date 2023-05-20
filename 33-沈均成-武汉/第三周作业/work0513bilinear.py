import cv2
import numpy as np


def bilinear(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]            #[0]是横向，是列数，[1]是竖向，是行数

    if src_h == dst_h and src_w == dst_w:            #如果原图像和目标图像尺寸相同，不需要插值，只用复制
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #通过目标图像坐标求出原图像坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找出用于插值的两个点，floor是向下取整函数，src_w - 1的目的是防止取的点在原图像之外
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算目标图像的值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


img = cv2.imread('lenna.png')
dst = bilinear(img, (700, 700))
cv2.imshow('bilinear interp', dst)
cv2.imshow("img", img)
cv2.waitKey()