# 用keras实现简单神经网络

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# step1. 构建一个3层神经网络，把模型结构存起来了，以及加入一些其他的东西
# 还没有真正的去算
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,))) # 激活函数设置成relu
network.add(layers.Dense(10, activation='softmax')) # --------- softmax，输出节点的激活函数是softmax，因为还要分类

# step2. compile(编译)，相当于把模型结构存起来，以及添加了一些额外的东西
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255


# 例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
print("before change:" , test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# 这里是真正的训练代码，只有一行
network.fit(train_images, train_labels, epochs=5, batch_size = 128)


'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
每个sample运行时间、loss、accuracy与什么有关？
- 速度：CPU/GPU
- loss、accuracy：数据的数量、质量
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss) 
print('test_acc', test_acc)

'''
生产环境推理：
输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break
