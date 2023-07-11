import cv2
import numpy as np

img = cv2.imread("lenna.png",1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst_gray = cv2.equalizeHist(img_gray)

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
dst_rgb = cv2.merge((rH,gH,bH))

cv2.imshow("image",np.hstack([img_gray,dst_gray]))
cv2.imshow("color_image",np.hstack([img,dst_rgb]))
cv2.waitKey(0)
