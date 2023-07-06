import cv2
import numpy as np

img = cv2.imread(r"F:\AI_Learn\data\lenna.png")  # 读图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图

# 特征点
sift = cv2.xfeatures2d.SIFT_create()  # 创建尺度不变特征变换对象的实例
keypoints, descriptor = sift.detectAndCompute(gray, None)  # 关键点和特征描述符

print(np.array(keypoints).shape)  # 关键点的形状/维数
print(descriptor.shape)  # 特征描述符的形状/维数
print(descriptor[0])  # 特征描述符的第一个维度的值

# 画图
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))
"""
 cv2.drawKeypoints()的5个参数
image:也就是原始图片
keypoints：从原图中获得的关键点，这也是画图时所用到的数据
outputimage：输出 //可以是原始图片
color：颜色设置，通过修改（b,g,r）的值,更改画笔的颜色，b=蓝色，g=绿色，r=红色。
flags：绘图功能的标识设置
flags的四个参数：

cv2.DRAW_MATCHES_FLAGS_DEFAULT：创建输出图像矩阵，使用现存的输出图像绘制匹配对和特征点，对每一个关键点只绘制中间点

cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：不创建输出图像矩阵，而是在输出图像上绘制匹配对

cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：对每一个特征点绘制带大小和方向的关键点图形

cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS：单点的特征点不被绘制
"""

# img=cv2.drawKeypoints(gray,keypoints,img) #默认参数
#
cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
