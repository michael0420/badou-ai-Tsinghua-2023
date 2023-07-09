import scipy.special
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivatives(x):
    # sigmoid导数
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)



class NeuralNetWork2:
    def __init__(self, input, bias1, bias2, weight1, weight2, learngin_rate, target):
        self.input = input
        self.bias1 = bias1
        self.bias2 = bias2
        self.weight1 = weight1
        self.weight2 = weight2
        self.target = target
        self.learning_rate = learngin_rate * 100


    def train(self, epoch):
        # 前向传播， 计算损失函数
        i = 1
        while i <= epoch:
            hidden = np.dot(self.weight1, self.input) + self.bias1
            hidden_sigmoid = sigmoid(hidden)
            # print(hidden_sigmoid)
            output = np.dot(self.weight2, hidden_sigmoid) + self.bias2
            output_sigmoid = sigmoid(output)
            loss = 0.5 * np.sum((output_sigmoid - self.target) ** 2)
            # print(loss)
            # 反向传播，计算梯度，更新权重
            # 更新隐藏层到输出层的权重
            second = (output_sigmoid - self.target) * sigmoid_derivatives(output)
            weight2_update = np.dot(np.expand_dims(second, axis=1), np.expand_dims(np.array(hidden_sigmoid), axis=1).T)

            # 更新输入层到隐藏层的权重
            hidden_update = np.dot(np.expand_dims(second, axis=1).T, self.weight2) # loss -> hidden
            # print(hidden_update)
            first = hidden_update * sigmoid_derivatives(hidden)
            # print(first)
            weight1_update = np.dot(first.T, np.expand_dims(np.array(self.input), axis=1).T)
            # print(weight1_update)
            self.weight2 -= learngin_rate * weight2_update
            self.weight1 -= learngin_rate * weight1_update

            print('第:{}次训练，loss：{}， '.format(i, loss))
            print('weight1:{},weight1_update:{}'.format(self.weight1, weight1_update))
            print('weight2:{},weight2_update:{}'.format(self.weight2, weight2_update))
            i += 1

    def query(self, inputs):
        hidden = np.dot(self.weight1, self.input) + self.bias1
        hidden_sigmoid = sigmoid(hidden)
        output = np.dot(self.weight2, hidden_sigmoid) + self.bias2
        return sigmoid(output)


if __name__ == '__main__':
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learngin_rate = 0.5
    # n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learngin_rate)
    # n.query([1.0, 0.5, -1.5])
    input = [0.05, 0.1]
    weight1 = [[0.15, 0.20], [0.25, 0.30]]
    weight2 = [[0.40, 0.45], [0.50, 0.55]]
    bias1 = 0.35
    bias2 = 0.60
    target = [0.01, 0.99]
    model = NeuralNetWork2(input, bias1, bias2, weight1, weight2, learngin_rate, target)
    model.train(10000)
    print(model.query(input))
