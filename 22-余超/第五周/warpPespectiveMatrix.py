# 透视变换 代码是利用源点与目标点坐标解出变换矩阵
# 参数：src（4*2矩阵） dst(4*2矩阵)  返回：透视变化的矩阵
# 属性 nums:源点个数 
import numpy as np
def warpPespectiveMatrix(src,dst):
    assert src.shape[0]=dst.shape[0] and src.shape[0]>=4
    nums = src[0]
    A=np.zeros((2*nums,1))   #写线性方程组左右矩阵A(2nums*8)、B(2nums*1) （A*warpMatrix=B)
    B=np.zeros((2*nums,1))
    for i in range(nums):  
        A[2*i,:]=[src[i,0],src[i,1],1,0,0,0,-1*src[i,0]dst[i,0],-1*src[i,1]*dst[i,0]]   #一行一行写
        A[2*i+1,:]=[0,0,0,src[i,0],src[i,1],1,-1*src[i,0]*dst[i,1],-1*src[i,1]*dst[i,1]]
        B[2*i,:]=dst[i,0]
        B[2*i+1,:]=dst[i,1]
A=np.mat(A) #mat生成的数组可以快速求得矩阵的逆
A=A.I
warpMatrix=np.dot(A,B)
#    还需要加上a33=1
warpMatrix=np.array(warpMatrix).reshape((8,0))
warpMatrix=np.insert(warpMatrix,warpMatrix.shape[0],1,1)
warpMatrix=warpMatrix.reshape((3,3))
return warpMatrix 
