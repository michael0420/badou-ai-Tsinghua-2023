import cv2
import numpy as np


# 画图函数，把两张图对应的关键点画出来，将两张图中的关键点用连线连接起来，以可视化显示匹配效果
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
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1 = cv2.imread("iphone1.png")
img2 = cv2.imread("iphone2.png")

# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()  # 尺度不变特征变换对象的实例
# sift = cv2.SURF()

# 根据两张图分别得到关键点和特征描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFmatcher with default parms

# 使用BFMatcher算法进行匹配，`cv2.BFMatcher(cv2.NORM_L2)`创建了一个Brute-Force  Matcher对象，
# 它使用L2范数（欧式距离）计算特征向量之间的距离
bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.knnMatch(des1, des2, k=2)  # `k  =  2`表示返回两个最佳匹配项

goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1, kp1, img2, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
