import numpy as np
import matplotlib.pyplot as plt
from typing import List


def normalization1(x: List[int]):
    """ 归一化1： y-min / max - min
    返回结果范围：（0～1）
    """
    if not x:
        return x
    min_v = min(x)
    max_v = max(x)
    ret = [(i - min_v) / (max_v - min_v) for i in x]
    return ret


def normalization2(x: List[int]):
    """ 归一化2 ： y- mean / max - min
    返回结果范围：（-1～1）
    """
    if not x:
        return x
    min_v = min(x)
    max_v = max(x)
    mean_v = np.mean(x)
    ret = [(i - mean_v) / (max_v - min_v) for i in x]
    return ret


def z_score(x: List[int]):
    """ 标准化 零均值化 ： y = (x - u) / sigma
    """
    if not x:
        return x
    # min_v = min(x)
    # max_v = max(x)
    mean_v = np.mean(x)
    s2 = sigma = sum([(i - mean_v) ** 2 for i in x]) / len(x)  # 标准差
    ret = [(i - mean_v) / sigma for i in x]
    return ret


if __name__ == '__main__':
    # x = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    #      11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

    x = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    n1 = normalization1(x)
    n2 = normalization2(x)
    zs = z_score(x)
    print(n1)
    print(n2)
    print(zs)
    n = []
    for i in x:
        n.append(x.count(i))  # 统计个数
    print(n)
    '''
    蓝线为原始数据，橙线为n2, 绿线为zs
    '''
    plt.plot(x, n)
    # plt.plot(n1, n)
    plt.plot(n2, n)
    plt.plot(zs, n)
    plt.show()
    pass
