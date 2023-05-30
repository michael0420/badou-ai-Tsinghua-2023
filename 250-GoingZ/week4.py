# 第四周作业
# 1.实现椒盐噪声
# 2.实现高斯噪声
# 3.实现PCA（detail）

import numpy as np
import cv2
from numpy import shape
import random

#### 1. 椒盐噪声 ####
def SaltNoise(percetage = 0.3):
    img = cv2.imread('lenna.png')
    NoiseNum = int(percetage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0]-1)
        randY = random.randint(0, img.shape[1]-1)
        if random.random() <= 0.5:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return img

#### 2. 高斯噪声 ####
def GaussianNoise(means = 0, sigma = 4, percetage = 0.3):
    img = cv2.imread('lenna.png')
    NoiseNum = int(percetage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0]-1)
        randY = random.randint(0, img.shape[1]-1)
        img[randX, randY] = img[randX, randY] + random.gauss(means, sigma)
        if img[randX, randY] < 0:
            img[randX, randY] = 0
        elif img[randX, randY] > 255:
            img[randX, randY] = 255
    return img

#### 3. PCA ####
class PCA():
    def __init__(self,n_components):
        self.n_components = n_components
    
    def fit_transform(self,X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T,X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals,eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:,idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X,self.components_)
 
# 调用
def PCA_test():
    pca = PCA(n_components=2)
    X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
    newX=pca.fit_transform(X)
    print(newX)                  #输出降维后的数据
