# -*- coding: utf-8 -

import numpy as np
import matplotlib.pyplot as plt

"""
最小二乘法的时间复杂度为，当 n 特别大的时候（一般大于 10000），求逆矩阵的过程非常复杂。此时采用最小二乘法，会非常耗时。
"""

def least_squares(x, y):
    n = len(x)  # 数据量
    sum_x = np.sum(x)  # x求和
    sum_y = np.sum(y)  # y求和
    sum_xy = np.sum(x * y)  # x*y的和
    sum_x2 = np.sum(x ** 2)  # x**2的平方和
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    return a, b  # 返回y = ax + b中的a和b


X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

a, b = least_squares(X, Y)
print(a)
print(b)
#画图
plt.scatter(X, Y)
plt.plot(X, a * X + b, color='red')
plt.show()
