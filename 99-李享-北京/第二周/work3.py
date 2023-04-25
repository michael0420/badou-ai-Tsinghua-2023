#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

# ��ȡ�Ҷ�ͼ��
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# �Ҷ�ͼ��ֱ��ͼ���⻯
dst = cv2.equalizeHist(gray)

# ֱ��ͼ
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
# ����ֱ��ͼ
plt.hist(dst.ravel(), 256)
plt.show()

# ��ʾԭʼ�Ҷ�ͼ��;���ֱ��ͼ���⻯���ͼ��
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

# ��ɫͼ��ֱ��ͼ���⻯

# ��ȡ��ɫͼ��
img = cv2.imread("lenna.png", 1)

# ��ʾԭʼ��ɫͼ��
cv2.imshow("src", img)

# �Բ�ɫͼ�����ֱ��ͼ���⻯����Ҫ�ֽ�ͨ������ÿһ��ͨ�����о��⻯
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# �ϲ�����ֱ��ͼ���⻯�����ÿ��ͨ��
result = cv2.merge((bH, gH, rH))

# ��ʾ����ֱ��ͼ���⻯�����Ĳ�ɫͼ��
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)