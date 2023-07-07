#! /usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd

data = pd.read_csv('/Users/aragaki/artificial/data/train_data.csv',sep='\s*,\s*',engine='python')
X = data['X'].values
Y = data['Y'].values

sumOfX = 0
sumOfY = 0
sumOfXY = 0
sumOfXX =0
num = 4

for i in range(num):
    sumOfX = sumOfX + X[i]
    sumOfXX = sumOfXX + X[i] * X[i]
    sumOfXY = sumOfXY + X[i] * Y[i]
    sumOfY = sumOfY + Y[i]

k = (num * sumOfXY- sumOfX * sumOfY)/(num * sumOfXX - sumOfX *sumOfX)
b = (sumOfY - k * sumOfX) / num
print("Coeff:{} Intercept: {}".format(k,b))
