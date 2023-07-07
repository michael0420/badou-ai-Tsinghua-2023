# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 10:09
# @Author  : zgh
# @FileName: network.py
# @Software: PyCharm
import scipy.special
import numpy


class Network:
	def __init__(self, inputnodes, hiddennodes, outputnodes, lr):
		# 初始化节点个数
		self.inputnodes = inputnodes
		self.hiddennodes = hiddennodes
		self.outputnodes = outputnodes
		self.lr = lr
		# 初始化全连接节点权重
		# 这里减去0.5是将随机初始的网络节点权重归到 -0.5 -- 0.5 之间，有正有负
		self.wih = numpy.random.rand(self.hiddennodes, self.inputnodes) - 0.5
		self.who = numpy.random.rand(self.outputnodes, self.hiddennodes) - 0.5
		# 初始化激活函数 scipy.special.expit(x)就是进行sigmoid:expit(x) = 1 / (1 + exp(-x))
		self.activation_fun = lambda x: scipy.special.expit(x)
		pass

	def train(self, inputs_list, targets_list):
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T

		i_h = numpy.dot(self.wih, inputs)
		i_h_a = self.activation_fun(i_h)
		h_o = numpy.dot(self.who, i_h_a)
		h_o_a = self.activation_fun(h_o)

		o_e = targets - h_o_a
		h_e = numpy.dot(self.who.T, o_e*h_o_a*(1-h_o_a))

		self.who += self.lr * numpy.dot((o_e*h_o_a*(1-h_o_a)),numpy.transpose(i_h_a))
		self.wih += self.lr * numpy.dot((h_e*i_h_a*(1-i_h_a)),numpy.transpose(inputs))
		pass

	def query(self, input):
		i_h = numpy.dot(self.wih, input)
		i_h_a = self.activation_fun(i_h)
		h_o = numpy.dot(self.who, i_h_a)
		h_o_a = self.activation_fun(h_o)
		#(h_o_a)
		return h_o_a

def get_dataset(root_path):
	data_file = open(root_path,'r')
	data_list = data_file.readlines()
	data_file.close()
	return data_list

def test():
	input_nodes = 10
	hidden_nodes = 5
	output_nodes = 3

	learning_rate = 0.3
	n = Network(input_nodes, hidden_nodes, output_nodes, learning_rate)
	n.query([1.0, 0.5, -1.5, 1.8, 1.4, -1.2, 0.6, -0.9, 1.3, -1.8])


if __name__ == '__main__':
	input_nodes = 28 * 28
	hidden_nodes = 200
	output_nodes = 10
	learning_rate = 0.15
	net = Network(input_nodes,hidden_nodes,output_nodes,learning_rate)
	# 读取的数据 [0] 是标签 [1:]是像素值拉伸结果 28*28
	train_data_path = 'dataset/mnist_train.csv'
	train_data = get_dataset(train_data_path)

	epochs = 5
	for i in range(epochs):
		for record in train_data:
			all_samples = record.split(',')
			# 像素值归一化
			inputs = (numpy.asfarray(all_samples[1:]))/255.0 * 0.99 +0.01
			targets = numpy.zeros(output_nodes)+ 0.01
			targets[int(all_samples[0])] = 0.99
			net.train(inputs,targets)


	#test()
	test_data_path = 'dataset/mnist_test.csv'
	test_data = get_dataset(test_data_path)
	scores = []
	for record in test_data:
		all_samples = record.split(',')
		# 像素值归一化
		gt = int(all_samples[0])
		print("该图形正确标签为",gt)

		inputs = (numpy.asfarray(all_samples[1:])) / 255.0 * 0.99 + 0.01
		outputs = net.query(inputs)
		label = numpy.argmax(outputs)
		print("该图片预测结果为",label)
		print("--------------------------next----------------------------")
		if label == gt:
			scores.append(1)
		else:
			scores.append(0)
	print(scores)

	# 计算图片判断的成功率
	scores_array = numpy.asarray(scores)
	print("perfermance = ", scores_array.sum() / scores_array.size)

