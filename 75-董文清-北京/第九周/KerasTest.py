from tensorflow.keras.datasets import mnist
#1、将训练集 和 测试集引入
(trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()
#2、展示测试集第一张图片（可选）
import matplotlib.pyplot as plt
firstTest = testImages[0]
plt.imshow(firstTest, cmap=plt.cm.binary) #展示二值图像
plt.show()

#3、搭建神经网络
from tensorflow.keras import models
from tensorflow.keras import layers
#连接每一层网络
network = models.Sequential()
#加层 512个神经元、激活函数relu
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
#输出层 10个神经元
network.add(layers.Dense(10, activation='softmax'))
#编译网络
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#4、处理训练数据和测试数据
#降维 + 归一化
trainImages = trainImages.reshape((60000, 28 * 28))
trainImages = trainImages.astype('float32') / 255
testImages = testImages.reshape((10000, 28 * 28))
testImages = testImages.astype('float32') / 255
#优化标记 one hot
from tensorflow.keras.utils import to_categorical
trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)
#5、进行训练（带输出日志）
network.fit(trainImages, trainLabels, batch_size=128, epochs=5)
#6、输入测试数据，查看准确度
testLoss, testAcc = network.evaluate(testImages, testLabels, verbose=1)
print("testLoss:", testLoss)
print("testAcc:", testAcc)
#7、输入一张手写图片，查看识别效果
(trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()
plt.imshow(testImages[1], cmap=plt.cm.binary)
plt.show()

res = network.predict(testImages.reshape((10000, 28 * 28)))
for i in range(res[1].shape[0]):
    if res[1][i] == 1 :
        print("the number of this picture is :", i)
        break



