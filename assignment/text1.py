"""调包实现RGB图片灰度化"""
from skimage.color import rgb2gray
import cv2

img = cv2.imread("lenna.png")
img_gray = rgb2gray(img)#掉包直接实现RGB转gray
cv2.imshow("img-gary",img_gray)
cv2.waitKey(0)
