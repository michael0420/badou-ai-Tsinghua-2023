# -*- coding: utf-8 -*-

# 归一化：
# 1. 减去最小值值求平均法
# 2. 减去平均值值求平均法
# 3. z-score标准化

import numpy as np
import matplotlib.pyplot as plt

# 1. 减去最小值值求平均法
# 结果：[0, 1]
# 公式：x_=(x−x_min)/(x_max−x_min)
def normalization1(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]


# 2. 减去平均值值求平均法
# 结果：[-1, 1]
# 公式：x_=(x−x_mean)/(x_max−x_min)
def normalization2(x):
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]


# 3. z-score标准化
# 结果：[-1, 1]
# 公式：x∗=(x−μ)/σ
def z_score(x):
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * ( i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean )  /s2 for i in x]


l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
cs=[]
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = normalization2(l) # TODO: 两种归一化都是-1到1？
z = z_score(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l,cs)
plt.plot(z,cs)
plt.show()
