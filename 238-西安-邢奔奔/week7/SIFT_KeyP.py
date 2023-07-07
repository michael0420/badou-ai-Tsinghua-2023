#! /usr/bin/python
# -*- coding:utf-8 -*-


import cv2
import numpy as np

img = cv2.imread('/Users/aragaki/artificial/image/lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray,None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))


#img =cv2.drawKeypoints(gray,keypoints,img)
cv2.imshow('sift_keypoints',img)
cv2.waitKey(0)
cv2.destroyWindow()