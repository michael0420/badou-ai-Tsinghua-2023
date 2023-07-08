import numpy as np
import scipy.special

class NeuralNetWork:
    #初始化函数
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):

        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.lr = learningRate
        #生成 均值为 0， 标准差为 sqrt(隐藏/输出的神经元节点个数）的权重矩阵
        #输入层到隐藏层
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        #隐藏层到输出层
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #设置激活函数为 sigmod
        self.activationFunction = lambda x : scipy.special.expit(x)

        pass
    #正向查询函数
    def query(self, inputs):
        #正向点积 计算
        hiddenInput = np.dot(self.wih, inputs)
        hiddenOutput = self.activationFunction(hiddenInput)
        finalInput = np.dot(self.who, hiddenOutput)
        finalOutput = self.activationFunction(finalInput)
        print(finalOutput)
        return finalOutput
    #训练函数
    def train(self, inputList, targetList):

        #压缩 输入 和 目标 为 二维矩阵并进行转置(反向传播所以要转置)
        inputs = np.array(inputList, ndmin=2).T
        targets = np.array(targetList, ndmin=2).T
        hiddenInput = np.dot(self.wih, inputs)
        hiddenOutput = self.activationFunction(hiddenInput)
        finalInput = np.dot(self.who, hiddenOutput)
        finalOutput = self.activationFunction(finalInput)

        #计算误差
        outputErrs = targets - finalOutput
        hiddenErrs = np.dot(self.who.T, outputErrs * finalOutput * (1 - finalOutput))
        self.who += self.lr * (np.dot(outputErrs * finalOutput * (1 - finalOutput), np.transpose(hiddenOutput)))
        self.wih += self.lr * (np.dot(hiddenErrs * hiddenOutput * (1 - hiddenOutput), np.transpose(inputs)))

        pass

#调试函数
def test():

    #读入训练数据
    trainDataFile = open('dataset/mnist_train.csv', 'r')
    trainDataList = trainDataFile.readlines()
    trainDataFile.close()
    #创建神经网络
    inputNodes, hiddenNodes, outputNodes, learningRate = 784, 300, 10, 0.1
    network = NeuralNetWork(inputNodes, hiddenNodes, outputNodes, learningRate)
    epochs = 5 #训练次数
    for i in range(epochs):
        for record in trainDataList:
            vals = record.split(',') #清理出数据 第一个元素是目标值
            #做归一化处理
            inputs = (np.asfarray(vals[1:])) / 255.0 * 0.99 + 0.01
            targets = np.zeros(outputNodes) + 0.01
            targets[int(vals[0])] = 0.99
            network.train(inputs, targets)

    #读入测试数据
    scores = []
    testDataFile = open('dataset/mnist_test.csv', 'r')
    testDataList = testDataFile.readlines()
    testDataFile.close()
    for record in testDataList:
        vals = record.split(',')
        correctNumber = int(vals[0])
        print("图片中数字为:", correctNumber)
        inputs = (np.asfarray(vals[1:])) / 255.0 * 0.99 + 0.01
        outputs = network.query(inputs)
        label = np.argmax(outputs)
        print("网络中认为的数字为:", label)
        scores.append(1 if label == correctNumber else 0)
    print(scores)
    scoresArray = np.asarray(scores)
    print("accuracy :", scoresArray.sum() / scoresArray.size)

if __name__ == '__main__':

    test()