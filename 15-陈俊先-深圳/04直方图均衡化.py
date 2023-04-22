import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("lenna.png", 0)
dst = cv2.equalizeHist(img)
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([img, dst]))
cv2.waitKey(0)

img2 = cv2.imread("lenna.png", 1)
cv2.imshow("src", img2)


(b, g, r) = cv2.split(img2)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst", result)
cv2.waitKey(0)
