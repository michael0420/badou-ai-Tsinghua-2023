import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.color import rgb2gray


mpl.use('TkAgg')

# 灰度化
img = cv.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(img_gray)
# print("image show gray: %s"%img_gray)
cv.imshow("image show gray",img_gray)

# 灰度化（调用）
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)   #原图

plt.subplot(222)
img_gray = rgb2gray(img)
plt.imshow(img_gray, cmap='gray')  #灰度化
# print("---image lenna----")
# print(img)

# 二值化
plt.subplot(223)
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.imshow(img_binary, cmap='gray')

#OTSU二值化
# plt.subplot(224)
# img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# ret2,mask_OTSU=cv.threshold(img, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# # print("OTSU的shape: ", mask_OTSU.shape)
# plt.imshow(mask_OTSU, cmap='gray')
# # plt.title("OTSU")

#解决imshow未响应
cv.waitKey()
cv.destroyAllWindows()
plt.show()