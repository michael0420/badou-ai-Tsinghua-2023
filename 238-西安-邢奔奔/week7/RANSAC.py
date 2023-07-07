#! /usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import scipy as sp
import scipy.linalg as sl

def Ransac(data, model, n, k, t, d, debug=False, Return_all=False):
    '''
    :param data: 数据集
    :param model: 已知的参数化模型
    :param n: 生成模型所需要的最少样本点
    :param k:最大迭代次数
    :param t:作为点满足模型条件的阈值
    :param d:拟合较好时需要的样本点的阈值
    :param debug:用于指示是否输出调试信息的布尔值，默认是False
    :param Return_all:用于指示是否输出所有迭代中的模型和内点的布尔值，默认是False
    :return:返回值为最佳模型和内点
    '''
    iterator = 0
    bestfit = None
    best_err = np.inf
    best_fit_index = None
    while iterator < k:
        maybe_idx, test_idx = random_partion(n,data.shape[0])
        print("maybe_idx:",maybe_idx)
        maybe_inliner = data[maybe_idx]
        test_data = data[test_idx]
        maybemodel = model.fit(maybe_inliner)
        test_err = model.geterror(test_data,maybemodel)
        print("test_err:",test_err < t)
        also_idx = test_idx[test_err < t]  #这里误差小于阈值的下标，意味着这些数据也是内群数据
        print("alse_idx:",also_idx)
        also_data = data[also_idx]
        if debug:
            print("test_error_min:", test_err.min())
            print("test_error_max:", test_err.max())
            print("numpy_mean(test_error):", np.mean(test_err))
            print("iteration %d,len %d",iterator,len(also_data))
        if(len(also_data) > d):
            better_data = np.concatenate((maybe_inliner,also_data))
            better_model = model.fit(better_data)
            better_errs = model.geterror(better_data,better_model)
            this_err = np.mean(better_errs)
            if this_err < best_err:
                best_err =this_err
                bestfit = better_model
                best_fit_index = np.concatenate((maybe_idx,also_idx))

        iterator += 1

    if bestfit == None:
        raise ValueError("don't meet bset fit model")
    if Return_all:
        return bestfit,{"inliners":best_fit_index}
    else:
        return bestfit
def random_partion(n,size):
    '''

    :param n: 需要产生的内点个数
    :param size: 数据点的个数
    :return: 假定的内群数据和测试数据的X坐标
    '''
    scale = np.arange(size)
    np.random.shuffle(scale)
    maybe_idx = scale[:n]
    test_idx  = scale[n:]
    return maybe_idx,test_idx

class LeastSquarefitModel:
    def __init__(self, input_colunms, output_colunms, debug=False):
        self.imput_colunm = input_colunms
        self.output_colunm = output_colunms
        self.debug = debug

    def fit(self,data):
        X = np.vstack([data[:,i] for i in self.imput_colunm]).T
        Y = np.vstack([data[:,i] for i in self.output_colunm]).T
        x, residu, rank, s = sl.lstsq(X, Y)
        '''
        scipy.linalg.lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True)
        a 是形状为 (M, N) 的系数矩阵，M 表示样本数量，N 是参数的数量。
        b 是形状为 (M,) 或 (M, K) 的依变量向量或矩阵。
        cond 是一个可选参数，表示奇异值分解的截断阈值。
        overwrite_a 和 overwrite_b 是可选参数，表示是否覆盖输入的数组 a 和 b。
        check_finite 是一个可选参数，表示是否检查数组中的无穷值和 NaN。
        函数返回一个包含以下信息的元组 (x, residuals, rank, s)：
        x 是最小二乘解。
        residuals 是残差的 Frobenius 范数。
        rank 是系数矩阵 a 的秩。
        s 是系数矩阵 a 的奇异值。
        '''
        return x


    def geterror(self,test_data,model):
        X = np.vstack([test_data[:,i] for i in self.imput_colunm]).T
        Y = np.vstack([test_data[:,i] for i in self.output_colunm]).T
        Y_fit = sp.dot(X,model)
        errors = np.sum((Y_fit - Y)**2,axis = 1)
        return errors

def test():
    n_sample = 500
    input_n = 1
    output_n = 1
    X = np.random.random((n_sample,input_n)) * 20
    perfect_fit = 60 * np.random.normal(size = (input_n,output_n))
    Y_exact = sp.dot(X, perfect_fit)

    X_noise = X + np.random.normal(size = (X.shape))
    Y_noise = Y_exact + np.random.normal(size = Y_exact.shape)

    if 1:
        n_outliners = 100
        all_idx = np.arange(X_noise.shape[0])
        np.random.shuffle(all_idx)
        outliner_idx = all_idx[:n_outliners]
        X_noise[outliner_idx] = 20 * np.random.random((n_outliners, input_n))
        Y_noise[outliner_idx] = 50 * np.random.normal(size = (n_outliners,output_n))

    data_all = np.hstack((X_noise, Y_noise))
    input_columns = range(input_n)
    output_columns = [input_n + i for i in range(output_n)]
    debug = False
    model = LeastSquarefitModel(input_columns,output_columns,debug)

    Liner_fit,resids,rank,s = sp.linalg.lstsq(data_all[:,input_columns], data_all[:,output_columns])
    ransac_fit, ransac_data = Ransac(data_all, model, 50,1000,7e3,300,debug= debug,Return_all=True)

    if 1:
        import pylab
        sort_idxs = np.argsort(X[:,0])
        X_sorted = X[sort_idxs]
        if 1:
            pylab.plot(X_noise[:,0],Y_noise[:,0],'k',label='data')
            pylab.plot(X_noise[ransac_data['inliners'],0],Y_noise[ransac_data['inliners'],0],'rx',label="Ransac_data")
        else:
            pylab.plot(X_noise[non_outliner_idx,0],Y_noise[no_outliner_idx,0],'k.',label='noisy_data')
            pylab.plot(X_nosie[outliner_idx,0],Y_noise[outliner_idx,0],'r',label='outliner_data')
        pylab.plot(X_sorted[:,0],np.dot(X_sorted,ransac_fit),label='ransac_fit')
        pylab.plot(X_sorted[:,0],np.dot(X_sorted,perfect_fit),label='perfect_fit')
        pylab.plot(X_sorted[:,0],np.dot(X_sorted,Liner_fit),label='Liner_fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()




