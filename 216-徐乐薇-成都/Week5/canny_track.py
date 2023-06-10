import cv2
import numpy as np

def CannyTreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold*ratio, apertureSize=kernel_size) # apertureSize默认为3
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  #用原始颜色增加到检测的边缘上
    cv2.imshow('canny demo', dst)

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #转换为灰度图像
cv2.namedWindow('canny demo')

#设置调节杠
'''
第一个参数：trackbar对象的名字
第二个参数：trackbar对象所在窗口的名字
第三个参数：trackbar对象的默认值
第四个参数：trackbar对象的最大值
第五个参数：回调函数，当trackbar对象的值发生改变时，调用回调函数
'''
cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyTreshold)

CannyTreshold(0)  # initialization
if cv2.waitKey(0) == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
