import numpy as np

'''
1、神经网络大概：需要初始化输入、隐藏、输出层节点数，初始化激活函数，初始化权重矩阵；
   含有两个函数，推理（query）和训练（train）
'''
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) -> None:
        # 初始化输入、隐藏、输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 初始化学习率
        self.lr = learningrate

        # 初始化权重矩阵范围[-0.5, 0.5]
        # wih表示的是输入层->隐藏层的权重矩阵
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # who表示的是隐藏层->输出层的权重矩阵
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        # 初始化激活函数
        # scipy.special.expit对应的是sigmoid函数
        import scipy.special
        # 我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, ):
        pass
    def query(self, ):
        pass
'''
2、推理过程：直接代入公式，加权求和
'''
def query(self, inputs):
    # inputs为输入矩阵
    # 首先计算出隐藏层的输入信号
    hidden_inputs = np.dot(self.wih, inputs)
    # 通过激活函数计算出隐藏层的输出信号
    hidden_outputs = self.activation_function(hidden_inputs)
    # 再计算输出层的输入信号
    final_inputs = np.dot(self.who, hidden_outputs)
    # 最后的输出信号
    final_outputs = self.activation_function(final_inputs)
    return final_outputs

