import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
b, g, r = cv2.split(img)
img_rgb = cv2.merge((r, g, b))  # r b 对调
# img_rgb2 = plt.imread("test.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度化
# print(img_gray, 'img_gray')
# cv2.imshow("img_gray", img_gray)
# cv2.waitKey()

_, img_bw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)    # 二值化
# print(_, "_")
# print(img_bw, "img_bw")
# cv2.imshow("img_bw", img_bw)
# cv2.waitKey()

plt.subplot(221), plt.imshow(img), plt.title("Original-bgr")
plt.subplot(222), plt.imshow(img_rgb), plt.title("Original-rgb")
plt.subplot(223), plt.imshow(img_gray, "gray"), plt.title("Gray")
plt.subplot(224), plt.imshow(img_bw, "gray"), plt.title("Black-White")
plt.show()
