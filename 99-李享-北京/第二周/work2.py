import numpy as np
import cv2

'''
python实现的双线性插值算法
'''


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]

    # 打印原始图像和目标图像的尺寸
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)

    # 如果尺寸相同，则直接复制图像
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # 创建一个空的目标图像
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    # 计算缩放因子
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    # 遍历目标图像的通道和像素
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 计算目标图像像素在原图像中的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找到用于计算插值的原图像中的点的坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算双线性插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    # 读取名为“lenna.png”的图像文件
    img = cv2.imread('lenna.png')

    # 使用双线性插值算法调整图像尺寸
    dst = bilinear_interpolation(img, (700, 700))

    # 显示调整后的图像
    cv2.imshow('bilinear interp', dst)

    # 等待按键关闭窗口
    cv2.waitKey()