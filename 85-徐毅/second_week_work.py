from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化方法一
img: object = cv2.imread("lenna.png")
h, w = img.shape[0:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)    #浮点算法
        # img_gray[i, j] = int(m[0]/3 + m[1]/3 + m[2]/3)            #平均值取法
        # img_gray[i, j] = int(m[1])                                #只取绿色
print(img_gray)
print(f"image show gray:{img_gray}")
print("--------------")
print(type(img_gray))
print("--------------")
cv2.imshow("image show gray", img_gray)
cv2.waitKey(0)

plt.subplot(221)   # 将窗口分为两行两列，当前位置为1
img = plt.imread("lenna.png")   # PNG 图像以浮点数组 (0-1) 的形式返回
print("----image lenna----")
print(img)

# 灰度化方法2
img_gray = rgb2gray(img)

# 灰度化方法3
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img_gray =img

plt.subplot(222)
plt.imshow(img_gray, cmap="gray")    # cmap表示绘制gray灰度图
print("---image gray----")
print(img_gray)

# 二值化方法1
rows, cols = img_gray.shape
# img_binary = np.zeros([rows, cols], img.dtype)
img_binary = img_gray
for i in range(rows):
    for j in range(cols):
        if (img_binary[i, j] <= 0.5):
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 1

# 二值化方法2
# img_binary = np.where(img_gray >= 0.5, 1, 0)

print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap = "gray")
plt.show()