import cv2
import numpy as np

def near_interp(img, size):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = size[0], size[1]
    if src_h == dst_h and src_w == src_w:
        return img.copy()
    targetImage = np.zeros((dst_h, dst_w, channels), np.uint8)
    h_plus = dst_h / src_h
    w_plus = dst_w / src_w
    for i in range(size[0]):
        for j in range(size[-1]):
            x = int(i / h_plus + 0.5)
            y = int(j / w_plus + 0.5)
            targetImage[i, j] = img[x, y]
    return targetImage

def bilinear_interp(img, size):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = size[0], size[1]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    targetImage = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)
    h_scale = float(src_h) / dst_h
    w_scale = float(src_w) / dst_w
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * w_scale - 0.5
                src_y = (dst_y + 0.5) * h_scale - 0.5

                src_x1 = int(np.floor(src_x))
                src_x2 = min(int(src_x1 + 1), src_w - 1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(int(src_y1 + 1), src_h - 1)

                temp1 = (src_x2 - src_x) * img[src_y1, src_x1, i] + (src_x - src_x1) * img[src_y1, src_x2, i]
                temp2 = (src_x2 - src_x) * img[src_y2, src_x1, i] + (src_x - src_x1) * img[src_y2, src_x2, i]
                targetImage[dst_y, dst_x, i] = int((src_y2 - src_y) * temp1 + (src_y - src_y1) * temp2)
    return targetImage



img = cv2.imread("lenna.png")
dst_size = [800, 800]
near_img = near_interp(img, dst_size)
bilinear_img = bilinear_interp(img, dst_size)
cv2.imshow("image", img)
cv2.imshow("near_interp", near_img)
cv2.imshow("bilinear_interp", bilinear_img)
cv2.waitKey(0)
