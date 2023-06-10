import cv2
import numpy as np

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 100, 200)) #第一个参数是灰度图像，第二个参数是阈值1，第三个参数是阈值2
cv2.waitKey(0)
cv2.destroyAllWindows()