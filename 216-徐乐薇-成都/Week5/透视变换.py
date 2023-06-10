import cv2
import numpy as np

img = cv2.imread("photo.jpg")
img_trans = img.copy()

# 输入输出均为图像对应的顶点坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]]) # 透视变换前的四个点.np.array也可以,但建议根据输入数组的类型正确选择所需的函数来避免出现错误。
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]]) # 透视变换后的四个点
print (img.shape)
#生成透视变换矩阵；进行透视变换
M = cv2.getPerspectiveTransform(src, dst) # 生成透视变换矩阵
print ("warpmatrix:")
print (M)

result = cv2.warpPerspective(img_trans, M, (337, 488)) # 进行透视变换,第一个参数是原始图像，第二个参数是变换矩阵，第三个参数是变换后的图像大小
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)