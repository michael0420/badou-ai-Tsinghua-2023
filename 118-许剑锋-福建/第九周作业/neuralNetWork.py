import scipy.special
import numpy as np

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 权重初始化
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        # 激活函数
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self, input, target, epoch):
        input = np.array(input, ndmin=2).T
        target = np.array(target, ndmin=2).T
        idx = 1
        while idx <= epoch:
            # 前向传播
            hidden_input = np.dot(self.wih, input)
            hidden_out = self.activation_function(hidden_input)
            final_input = np.dot(self.who, hidden_out)
            final_out = self.activation_function(final_input)
            # 计算误差
            error = target - final_out
            # 反向传播
            hidden_error = np.dot(self.who.T, (error * final_out * (1 - final_out)))
            self.who += self.lr * np.dot((error * final_out * (1 - final_out)), np.transpose(hidden_out))
            self.wih += self.lr * np.dot((hidden_error * hidden_out * (1 - hidden_out)), np.transpose(input))
            idx += 1
            print(idx, error)


    def query(self, inputs):
        hidden_input = np.dot(self.wih, inputs)
        hidden_out = self.activation_function(hidden_input)
        final_input = np.dot(self.who, hidden_out)
        final_out = self.activation_function(final_input)
        return final_out


if __name__ == '__main__':
    input_nodes = 2
    hidden_nodes = 2
    output_nodes = 2
    learngin_rate = 0.5
    # n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learngin_rate)
    # n.query([1.0, 0.5, -1.5])
    input = [0.05, 0.1]
    weight1 = [[0.15, 0.20], [0.25, 0.30]]
    weight2 = [[0.40, 0.45], [0.50, 0.55]]
    bias1 = 0.35
    bias2 = 0.60
    target = [0.01, 0.99]
    model = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learngin_rate)
    model.train(input, target, 10000)
    print(model.query(input))
