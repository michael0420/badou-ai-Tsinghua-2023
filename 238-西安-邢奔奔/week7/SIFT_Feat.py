#! /usr/bin/python
# -*-coding:utf-8 -*-

import cv2
import numpy as np

def drawMatches_cv2(img1,kp1,img2,kp2,goodMatches):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1,:w1] = img1
    vis[:h2,w1:w1+w2] = img2

    p1 = [kpp.queryIdx for kpp in goodMatches]
    p2 = [kpp.trainIdx for kpp in goodMatches]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2])+(w1,0)

    for (x1,y1), (x2,y2) in zip(post1,post2):
        cv2.line(vis, (x1,y1), (x2,y2), (0,0,255))

    cv2.namedWindow("match",cv2.WINDOW_AUTOSIZE)

    cv2.imshow("match",vis)


img1 = cv2.imread('/Users/aragaki/artificial/image/iphone1.png')
img2 = cv2.imread('/Users/aragaki/artificial/image/iphone2.png')

sift = cv2.xfeatures2d.SIFT_create()
# 4.0以上版本
# sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_L2,False)

matches = bf.knnMatch(des1,des2,2)
# DMatch 对象具有以下属性：
#
# DMatch.queryIdx：查询图像中关键点的索引。
# DMatch.trainIdx：训练图像中关键点的索引。
# DMatch.distance：关键点之间的距离，用于表示它们之间的相似度或差异度。距离值越小，表示两个关键点越相似或匹配度越高。

goodmatches = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        goodmatches.append(m)

drawMatches_cv2(img1,kp1,img2,kp2,goodmatches[:20])
cv2.waitKey(0)
cv2.destroyAllWindow()