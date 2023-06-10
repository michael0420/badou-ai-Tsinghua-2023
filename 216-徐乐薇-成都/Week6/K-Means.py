import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''
# 第一步确定K值，将数据集聚集成K个类簇或小组
# 第二步确定K个初始质心，随机选择K个质心
# 第三步计算每个点到质心的距离，将每个点分配到最近的质心
# 第四步重新计算每个簇的质心，计算每个簇中所有点的均值（中心点）。虚拟点的坐标为每个组所有点X和Y的均值
# 第五步重复第三步和第四步，直到质心不再变化

# 读取图片灰度值
img = cv2.imread('lenna.png', 0) # 0表示灰度图像
print (img.shape) # (512, 512)

# 获取图像高度和宽度
rows, cols = img.shape[:2]

# 图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon) type:终止的类型 max_iter:最大迭代次数 epsilon:精确度
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # 10次迭代或者移动距离小于1.0

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS # 随机初始中心

# K-Means聚类 聚集成4类
# compactness：紧密度，返回每个点到相应重心的距离的平方和
# labels：标志数组（与上一节提到的代码相同），每个成员被标记为0，1，2等
# centers：由聚类的中心组成的数组
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags) # 4类

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1])) # 根据labels的形状重新整理成与img相同的形状

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来设置Matplotlib库的字体参数，以便它可以在图表中正确显示中文字符。

# 显示图像
titles = [u'原始图像', u'聚类图像'] # 图像标题
images = [img, dst] # 图像list 用于for循环 一次显示两张图像
for i in range(2): # 取值为0,1
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'), plt.title(titles[i]) # 1行2列 i+1表示第几个
    plt.xticks([]), plt.yticks([]) # 隐藏坐标轴
plt.show()