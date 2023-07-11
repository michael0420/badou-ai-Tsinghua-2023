# -*- coding: utf-8 -*-

# 从零开始实现神经网络：调用
# 图片和excel文件未传

import numpy
import matplotlib.pyplot as plt
from index import NeuralNetwork

data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28)) # 第一个是正确答案，去掉

onodes = 10 # 最外层有10个输出节点
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)


input_nodes = 784
hidden_nodes = 100 # 100是经验值，实验科学
output_nodes = 10
learning_rate = 0.3
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
training_data_file = open("dataset/mnist_train.csv")
trainning_data_list = training_data_file.readlines()
training_data_file.close()
for record in trainning_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)


# 训练循环10次
epochs = 10
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in trainning_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


scores = []
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 归一化
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 推理，没有进行softmax，直接用numpy.argmax进行排序，选择概率最大的推理结果
    # 如果想获取概率的 value，需要选择softmax
    outputs = n.query(inputs)
    label = numpy.argmax(outputs) # numpy.argmax 返回的是最大值的索引
    print("output reslut is : ", label) # 巧了，这里的索引和实际预测的数字相等
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
scores_array = numpy.asarray(scores) # 成功率
print("perfermance = ", scores_array.sum() / scores_array.size)
