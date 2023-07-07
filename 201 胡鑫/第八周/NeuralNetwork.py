import numpy as np
import scipy.special

class Neuralnetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) -> None:
        # 初始化输入、隐藏、输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 初始化学习率
        self.lr = learningrate
        # 初始化权重矩阵
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        # self.wih = (np.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes) )  )
        # self.who = (np.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes) )  )
        # 初始化激活函数sigmoid
        self.activation_function = lambda x:scipy.special.expit(x)    

    def train(self, inputs, targets):
        # inputs和targets分别为输入的训练数据和对应的结果
        # 为了后面的矩阵运算，先将格式整理，转换成列向量

        '''
        在 train 函数中保留 inputs = np.array(inputs, ndmin=2).T 这一行是必要的，
        以确保 inputs 是一个列向量，以便进行后续的矩阵计算。而在 query 函数中，
        由于不涉及到更新权重矩阵的操作，inputs 是一个一维数组的情况下也可以正常执行（numpy会自动进行广播操作）。
        '''
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        # 后面几步与推理一样
        # 计算到达隐藏层的输入信号
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层的输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层的输入信号
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        # 输出层误差
        output_errors = (targets - final_outputs)
        # 隐藏层的误差（梯度）可以通过下一层（即输出层）的误差、当前层的权重矩阵的转置和当前层的激活函数的导数来计算
        # 这个计算过程实际上利用了链式法则来传播误差，并计算每一层对误差的贡献
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 更新权重值
        self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs), 
                                     hidden_outputs.T)
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)

    def query(self, inputs):
        # inputs为输入的信号，转化为列向量
        inputs = np.array(inputs, ndmin=2).T
        # 计算到达隐藏层的输入信号
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层的输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层的输入信号
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出信号
        final_outputs = self.activation_function(final_inputs)
        # print(final_outputs)
        return final_outputs
    
'''
    1、初始化神经网络，读取训练集，开始训练
'''
# 初始化神经网络
input_nodes = 28 * 28
hidden_nodes = 100
output_nodes = 10
learningrate = 0.3

netwk = Neuralnetwork(input_nodes, hidden_nodes, output_nodes, learningrate)

# 读取训练集
with open('../dataset/mnist_train.csv') as f:
    train_data_list = f.readlines()

# 设置网络的训练循环次数
epochs = 10
for e in range(epochs):
    # 数据预处理、设置对应结果以及训练
    for record in train_data_list:
        # csv文件，用逗号分开
        values = record.split(',')
        # 数据预处理，提取属于图像的数据，转换成ndarray型并归一化
        inputs = np.asfarray(values[1:]) / 255 * 0.99 + 0.01
        # 设置对应的正确结果
        targets = np.zeros(output_nodes) + 0.01
        targets[int(values[0])] = 0.99
        # 训练
        netwk.train(inputs, targets)

'''
    2、输入测试集，测试网络的识别能力
'''
# 读取测试集
with open('../dataset/mnist_test.csv') as f:
    test_data_list = f.readlines()

# 数据预处理，计算得分，评估能力
scores = []
for record in test_data_list:
    values = record.split(',')
    # 先输出正确答案
    correct = int(values[0])
    print("该图片的数字为 ", correct)
    # 数据预处理
    inputs = np.asfarray(values[1:]) / 255 * 0.99 + 0.01
    # 推理
    res = netwk.query(inputs)
    # 找到最大的输出值的下标位置
    label = np.argmax(res)
    print("网络认为此图片的数字为 ", label)
    if label == correct:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算识别成功率
nd_scores = np.asarray(scores)
print("成功率：", nd_scores.sum() / nd_scores.size)