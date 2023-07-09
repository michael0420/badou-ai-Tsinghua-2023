import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

[1]
#载入数据到内存中,由于版本较低，访问网络出错，故使用压缩包
path = r'mnist.npz'
with np.load(path) as f:
    train_images,train_lables = f['x_train'],f['y_train']
    test_images,test_lables = f['x_test'],f['y_test']
#打印数据的shape
print('train_images.shape:',train_images.shape)
print('train_lables:',train_lables)
print('test_images.shape:',test_images.shape)
print('test_lables:',test_lables)

[2]
#显示test_images的第一个图片

plt.imshow(test_images[0],cmap=plt.cm.binary)
plt.title('the fisrt test_image')
plt.show()

[3]
#创建一个串联全连接的人工神经网络模型
network = models.Sequential()
#添加隐藏层512，激活函数为relu，输入为28*28层
network.add(layers.Dense(512 , activation = 'relu' , input_shape = (28*28,)))
#添加输出层，1~10个数字所以10个输出
network.add(layers.Dense(10,activation = 'softmax'))
'''
编译模型
optimizer='rmsprop'：指定优化器为RMSprop，它是一种基于梯度的优化算法，用于优化模型的权重。
loss='categorical_crossentropy'：指定损失函数为分类交叉熵损失函数，它用于衡量模型预测结果与真实标签之间的差异。
metrics=['accuracy']：指定评估指标为准确率，它用于评估模型在预测类别时的性能。
'''
network.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])

[4]
#将数据降维并进行归一化
train_images = train_images.reshape(60000,28*28)
train_images = train_images.astype('float')/255.0

test_images = test_images.reshape(10000,28*28)
test_images = test_images.astype('float')/255.0

[5]
'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
'''
print('before:',test_lables[0])
train_lables = to_categorical(train_lables)
test_lables = to_categorical(test_lables)
print('after:',test_lables[0])

[6]
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images,train_lables,epochs = 5,batch_size = 128)

[7]
'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss,test_acc = network.evaluate(test_images,test_lables,verbose = 1)
print('test_loss:',test_loss)
print('test_acc:',test_acc)

[8]
'''
输入一张手写数字图片到网络中，看看它的识别效果
'''
#重新载入数据到内存中,由于版本较低，访问网络出错，故使用压缩包
path = r'mnist.npz'
with np.load(path) as f:
    train_images,train_lables = f['x_train'],f['y_train']
    test_images,test_lables = f['x_test'],f['y_test']
#显示要测试的图片test_images[1]
plt.imshow(test_images[1])
plt.show()
#测试图片转成2维
test_images = test_images.reshape(10000,28*28)
res = network.predict(test_images)
print(res)
#刚才显示的test_images[1]是否被识别
for i in range(res[1].shape[0]):
    if res[1][i] == 1:
        print('the number for the pictrue is :',i)
        break
