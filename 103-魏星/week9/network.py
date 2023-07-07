import numpy as np
from keras.datasets import fashion_mnist as mnist
import keras.utils as utils
import cv2

'''
准备数据
    此处使用Fashion-MNIST 数据集。
    服装类数据库，构成如下：
    图片：28*28，0～255灰度图，0-9手写数字
    标签：0-9 10个类别分别代表T恤、裤子、套头毛衣、裙子、外套、凉鞋、衬衫、运动鞋、包包及短靴
    训练集：60000张
    测试集：10000张
定义模型
    入参：输入、输出、控制条件(学习率、损失函数)
训练模型
    60000 = 10*6000
测试模型
    10000 = 10*1000
预测 
    
'''
class NetWork:
    def __init__(self,inputnodes, hidenodes, outnodes, learningrates):
        self.inodes = inputnodes
        self.hnodes = hidenodes
        self.onodes = outnodes

        # 权重初始化
        self.i2hweight = np.random.random((self.inodes,self.hnodes)) - 0.5
        self.h2oweight = np.random.random((self.hnodes,self.onodes)) - 0.5

        # 学习率
        self.lr = learningrates

        # 激活函数 tanh
        # self.activation_func = lambda x:(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        self.activation_func = lambda x:1/(1+np.exp(-x))

    def train(self,train_data,labels_data):
        # print("1-1:",self.i2hweight.shape)  # 784*150
        # print("1-2:",self.h2oweight.shape)  # 150*10
        # print("2-1:", train_data.shape)
        # print("2-2:", labels_data.shape)

        inputs = np.expand_dims(train_data, axis=0) #1*784
        targets = np.expand_dims(labels_data, axis=0) #1*10
        # print("2-1:", inputs.shape)
        # print("2-2:", targets.shape)

        hidden_input = np.dot(inputs, self.i2hweight) # 1*150
        hidden_output = self.activation_func(hidden_input) #1*150
        # print("3-1:", hidden_input.shape)
        # print("3-2:", hidden_output.shape)

        final_input = np.dot(hidden_output, self.h2oweight) #1*10
        final_output = self.activation_func(final_input) #1*10
        # print("4-1:", final_input.shape)
        # print("4-2:", final_output.shape)

        output_errors = targets - final_output #1*10
        hidden_errors = np.dot(self.h2oweight, output_errors.T*((1-final_output.T*final_output.T))) # 150*1
        # print("5-1:", output_errors.shape)
        # print("5-2:", hidden_errors.shape)

        self.h2oweight += self.lr * np.dot(np.transpose(hidden_output),
                                           (output_errors * final_output * (1-final_output))) # 150*10
        self.i2hweight += self.lr * np.dot(np.transpose(inputs),
                                           (hidden_errors.T * hidden_output* (1-hidden_output))) # 784*150

    def test(self,test_data):
        hidden_input = np.dot(test_data, self.i2hweight)
        hidden_output = self.activation_func(hidden_input)
        final_input = np.dot(hidden_output, self.h2oweight)
        final_output = self.activation_func(final_input)
        return final_output


def main():
    label_desc = ["T恤", "裤子", "套头毛衣", "裙子", "外套", "凉鞋", "衬衫", "运动鞋", "包包", "短靴"]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('train_images.shape=',x_train.shape)
    print('train_labels.shape=',y_train.shape)
    print('test_images.shape=',x_test.shape)
    print('test_labels.shape=',y_test.shape)

    train_images = x_train.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    train_labels = utils.to_categorical(y_train)

    test_images = x_test.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    # test_labels = utils.to_categorical(y_test)
    test_labels = y_test

    inputnodes = 28*28
    hidenodes = 150
    outnodes = 10 #10个标签
    learningrate = 0.1
    network = NetWork(inputnodes, hidenodes, outnodes, learningrate)

    epochs = 5
    for i in range(epochs):
        print("第",i,"代")
        for train_data,labels_data in zip(train_images,train_labels):
            network.train(train_data, labels_data)

    scores = []
    for test_data,test_label in zip(test_images,test_labels):
        print("该图片对应的数字为",test_label,",标签为：",label_desc[test_label])
        out_label = network.test(test_data)
        label = np.argmax(out_label)
        print("网络认为图片的数字是：",label,",标签为：",label_desc[label])
        if label == test_label:
            scores.append(1)
        else:
            scores.append(0)
    # print(scores)

    sum = np.sum(scores)
    print("准确率:", sum/np.size(scores))

    img1 = cv2.imread("./dataset/yuce.png",0)
    img1_input = img1.reshape(inputnodes)
    img1_out = network.test(img1_input)
    print("img1(短靴)被推断为：", label_desc[np.argmax(img1_out)])

    img2 = cv2.imread("./dataset/yuce2.png", 0)
    img2_input = img2.reshape(inputnodes)
    img2_out = network.test(img2_input)
    print("img2(套头毛衣)被推断为：", label_desc[np.argmax(img2_out)])


if __name__=='__main__':
    main()


