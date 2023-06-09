# -*- coding: utf-8 -*-
# @Time    : 2023/6/16 8:35
# @Author  : zgh
# @FileName: sift_match.py
# @Software: PyCharm


import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
	h1, w1 = img1_gray.shape[:2]
	h2, w2 = img2_gray.shape[:2]

	vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
	vis[:h1, :w1] = img1_gray
	vis[:h2, w1:w1 + w2] = img2_gray

	p1 = [kpp.queryIdx for kpp in goodMatch]
	p2 = [kpp.trainIdx for kpp in goodMatch]

	post1 = np.int32([kp1[pp].pt for pp in p1])
	post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

	for (x1, y1), (x2, y2) in zip(post1, post2):
		cv2.line(vis, (x1, y1), (x2, y2), (153, 204, 255))

	cv2.namedWindow("match", cv2.WINDOW_NORMAL)
	cv2.imshow("match", vis)

if __name__ == '__main__':
	img1 = cv2.imread("../img/flowers1.jpg")
	img2 = cv2.imread("../img/flowers2.jpg")
	sift = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)
	# knnMatch()是BFMatcher类中的一个函数，用于在两组特征描述子之间进行k近邻匹配。其中des1和des2分别表示两组特征描述子，
	# k表示每个特征点在另一幅图像中匹配时找到的最佳k个匹配结果，这里默认为2，即返回每个特征点的两个最佳匹配结果。
	bf = cv2.BFMatcher(cv2.NORM_L2)
	matches = bf.knnMatch(des1, des2, k=2)

	goodMatch = []
	# 取更近的那个匹配结果
	for m, n in matches:
		if m.distance < 0.6 * n.distance:
			goodMatch.append(m)

	drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch[:20])

	cv2.waitKey(0)
	cv2.destroyAllWindows()
