# -*- coding: utf-8 -*-
# 从零开始实现神经网络：定义NeuralNetwork类
# 图片和excel文件未传

import numpy
import scipy.special

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # 减去0.5是为了让权重变成[-0.5, 0.5]
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.inodes) - 0.5

        # 设置模型全局激活函数为 sigmoid 函数
        self.activation_function = lambda x: scipy.special.expit(x)

    # 推理函数，即训练函数的正向过程
    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # ! 这里得到的神经网络的计算结果，还没到分类
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs
    
    # 训练过程，根据输入的训练数据更新节点链路权重
    # 分为正向、反向两步
    def train(self, inputs_list, targets_list):
        # step1. 正向过程
        # 把inputs_list, targets_list转换成numpy支持的二维矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T # ndmin=2：指定target数组最小维度为2
        targets = numpy.array(targets_list, ndmin=2).T

        # 以下4行同query
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # step2. 反向传播
        # 先通过损失函数计算误差
        # 这里计算误差用了简单的减法，工程里可以用交叉熵、MSE等
        output_errors = targets - final_outputs 
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs  *(1 - final_outputs))

        # 根据误差进行反向传播，把更新加到原来链路的权重上（套用ppt上的公式）
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))

'''
# run test
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.query([1.0, 0.5, -1.5]) # [0.43001594 0.47261784 0.48035287]


# 使用实际数据来训练我们的神经网络，类似于load_data
data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
len(data_list)
data_list[0]
# 绘图
#把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
#第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))

# 这里的归一化技巧，调整取值范围：
# 需要归一化到[0.01, 1]之间
# 先乘以0.99是为了避免过小的数字直接除以255后得0
# 加上0.01是为了满足所有数据落在归一化区间
scaled_input = image_array / 255.0 * 0.99 + 0.01
'''
