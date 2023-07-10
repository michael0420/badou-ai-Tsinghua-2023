# -*- coding: utf-8 -*-
import numpy.random
import scipy.special
from tensorflow.keras.datasets import mnist


class NeuralNetWork:
    def __init__(self, input_nodes, hidde_nodes, output_nodes, learning_rate):
        """
        初始化神经网络
        :param input_nodes: 输入层节点数
        :param hidde_nodes: 隐藏层节点数
        :param output_nodes: 输出层节点数
        :param learning_rate: 学习率
        """
        self.input_nodes = input_nodes
        self.hidde_nodes = hidde_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 随机初始化权重值到[-0.5, 0.5]之间
        self.w_input_hidde = numpy.random.rand(self.hidde_nodes, self.input_nodes) - 0.5
        self.w_hidde_output = numpy.random.rand(self.output_nodes, self.hidde_nodes) - 0.5

        # 设置激活函数，此处使用sigmoid
        self.activation_function = scipy.special.expit
        return

    def query(self, inputs):
        """
        推理函数
        :param inputs: 输入待推理的数据
        :return: 推理结果
        """
        # 通过输入计算出隐藏层结果
        hidde_inputs = numpy.dot(self.w_input_hidde, inputs)
        # 调用激活函数
        hidde_outputs = self.activation_function(hidde_inputs)
        # 计算输出层结果
        final_result = numpy.dot(self.w_hidde_output, hidde_outputs)
        final_outputs = self.activation_function(final_result)
        return final_outputs

    def train(self, inputs, targets):
        # 通过输入计算出隐藏层结果
        hidde_inputs = numpy.dot(self.w_input_hidde, inputs)
        # 调用激活函数
        hidde_outputs = self.activation_function(hidde_inputs)
        # 计算输出层结果
        final_result = numpy.dot(self.w_hidde_output, hidde_outputs)
        final_outputs = self.activation_function(final_result)

        # 计算损失
        outputs_diff = targets - final_outputs
        # 计算隐藏层输出损失
        hidden_diff = numpy.dot(self.w_hidde_output.T, outputs_diff * final_outputs * (1 - final_outputs))
        # 根据损失更新权重
        self.w_hidde_output += self.learning_rate * numpy.dot(outputs_diff * final_outputs * (1 - final_outputs), numpy.transpose(hidde_outputs))
        self.w_input_hidde += self.learning_rate * numpy.dot(hidden_diff * hidde_outputs * (1 - hidde_outputs), numpy.transpose(inputs))


if __name__ == '__main__':
    # 加载数据集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # 输入层节点数=图片长*宽
    network = NeuralNetWork(28 * 28, 200, 10, 0.1)
    # 指定训练次数
    epochs = 3
    for epoch in range(epochs):
        for i in range(len(train_images)):
            targets = numpy.zeros(10)
            targets[train_labels[i]] = 1
            network.train(train_images[i].reshape((1, -1)).T / 255, targets.reshape((1, -1)).T)
    print('训练完成: ', network.w_input_hidde, network.w_hidde_output)

    # 使用测试集验证
    success_count = 0
    for i in range(len(test_images)):
        result = network.query(test_images[i].reshape((1, -1)).T / 255)
        # 返回最大值索引，由于训练的是数字识别，索引其实就是等于实际的值
        label = numpy.argmax(result)
        success_count += (1 if label == test_labels[i] else 0)
    print('成功率：', success_count / len(test_labels))
    # 输出 0.9625
