#pca主成分分析法，主要分为四步：1.去均值化  2.求得协方差矩阵 3.求得协方差矩阵的特征值以及对应特征向量 
# 4.根据需求按特征值由大至小排序提取前k个特征向量 5.依次按列排列此特征向量得到特征降维的线性变换，右乘原特征信息矩阵
import numpy as np
# 定义一个PCA的类
class PCA(object):
    def __init__(self,x,k):
#         x、k分别代表样本矩阵、需要降维的维度数
        self.x=x
        self.k=k
#         中心化后的样本矩阵
        self.center=[]
#     中心化后样本矩阵的协方差矩阵
        self.Cov=[]
#         降维矩阵
        self.convert=[]
#        样本矩阵中心化
        self.Z=[]
        self.center=self.centralized()
        self.Cov=self._cov()
        self.convert=self._U()
        self.Z=self._Z()
        
    def centralized(self):
       centra=[]
#  先按行计算样本均值矩阵mean，矩阵行数列数与样本矩阵相同
#  mean代表样本矩阵的均值,样本矩阵每行是一个样本，每列是一个维度，所以要按行取均值
# 矩阵.T返回矩阵的转置矩阵
      
#       mean = np.array(np.mean(atter for atter in self.x.T))
       n=np.shape(self.x)[0]
       mean=np.mean(self.x,1)
       means=np.array([mean for i in range(n)]).T
       centra=self.x-means
       return centra
# 求中心化矩阵的协方差矩阵
    def _cov(self):
        n1=np.shape(self.center)[0]
#         np.dot()-矩阵乘法、
        C=np.dot(self.center.T,self.center)/(n1-1)
        return C
# 求协方差矩阵的特征向量以及对应的特征值,按特征值从大至小相应特征值排列得到降维矩阵,shape为（m，k），m为原特征维度，k为降维后的维度
    def _U(self):
#       np.linalg.eig()计算矩阵的特征值与特征向量并返回对应数组
        a,b=np.linalg.eig(self.Cov)
#       np.argsort()将数组由小至大进行排序后并返回相应的索引的数组
        sort=np.argsort(-1*a)
#       将前k个特征向量排列得到所需的降维矩阵  
        U=[b[:,sort[i]] for i in range(self.k)]
#       此时U.shape为[k,m]需要转置一次
        D=np.array(U).T
        print('样本矩阵的降维矩阵D:\n',D)
        return D
    def _Z(self):
        Z=np.dot(self.x,self.convert)
        print('样本矩阵降维后的矩阵Z:\n',Z)
        return Z  
      
#       测试
if __name__ == '__main__':
    x=np.array([[1,4,6,2,3],
                [2,4,8,4,6],
                [7,4,3,8,9],
                [1,3,6,7,3],
                [3,5,2,8,5]])
    pca = PCA(x,3)
