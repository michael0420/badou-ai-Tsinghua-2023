import numpy as np

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0] # 4
    A = np.zeros((2*nums, 8)) # 8x8 根据变换方程的输入矩阵
    B = np.zeros((2*nums, 1)) # 8x1 根据变换方程的输出矩阵
    for i in range(nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2*i, :] = np.array([A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]) # 偶数行和左边元素计算x坐标方程式的系数
        A[2*i+1, :] = np.array([0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]) # 奇数行和右边元素计算y坐标方程式的系数
        B[2*i, :] = B_i[0] # 偶数行和左边元素计算x坐标方程式的常数项
        B[2*i+1, :] = B_i[1] # 奇数行和右边元素计算y坐标方程式的常数项
    A = np.mat(A) # 将数组或列表转换成矩阵 8x8

    #求出A的逆矩阵，与B相乘
    warpMatrix = A.I * B # 8x1，求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 将warpMatrix转换成3x3的矩阵
    # 因为OpenCV中的透视变换函数要求传入的仿射变换矩阵是一个3x3的矩阵
    warpMatrix = np.array(warpMatrix).T[0] # 将原始数组压缩成一个一维数组1x8
    warpMatrix = np.append(warpMatrix, 1) # 1x9 插入a_33 = 1
    warpMatrix = warpMatrix.reshape(3, 3) # 3x3
    return warpMatrix

if __name__ == '__main__':
    print('warpMatrix')
    src = [[12.0, 271.0], [30.0, 277.0], [164.0, 134.0], [11.0, 121.0]]
    src = np.array(src)  # 4x2
    dst = [[30.0, 48.0], [88.0, 72.0], [12.0, 15.0], [24.0, 21.0]]
    dst = np.array(dst)  # 4x2

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
