import matplotlib.pyplot as plt
import numpy
import cv2

img = cv2.imread("scb.jpg")
h, w = img.shape[:2]
img_g = numpy.zeros([h, w], img.dtype)
img_b = numpy.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_g[i, j] = (m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
        if img_g[i, j] > 128:
            img_b[i, j] = 255
        else:
            img_b[i, j] = 0

img_gray = cv2.cvtColor(img_g, cv2.COLOR_GRAY2RGB)
img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)


# img_binary = numpy.where(img_g >= 127, 1, 0)

plt.subplot(1, 2, 1)
plt.imshow(img_gray)
plt.title('Gray Image')

plt.subplot(1, 2, 2)
plt.imshow(img_b)
# plt.imshow(img_binary, cmap='gray')
plt.title('Binary Image')
plt.show()
