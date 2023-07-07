from keras.datasets import mnist
from keras import layers, models
from matplotlib import pyplot as plt
from keras.utils import to_categorical

'''
1 加载数据集（训练集和测试集）
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢)：
train_images是用于训练系统的手写数字图片；
train_labels是用于标注图片的信息；
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images.shape = ", train_images.shape)
print("train_labels = ", train_labels)
print("test_images.shape = ", test_images.shape)
print("test_labels = ", test_labels)

'''
2 查看一张测试图片
'''
# _ = test_images[0]
# plt.imshow(_, cmap=plt.cm.binary)
# plt.show()

'''
3 使用keras构建一个有效识别图像的神经网络
layers：表示神经网络中的一个数据处理层
models.Sequential：表示吧每一个数据处理层串联起来（dense：全连接层）
layers.Dense：构造一个数据处理层
input_shape(28*28,)：表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的','表示数组里面的每一个元素到底包含多少个数字都没有关系。
'''
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
# loss使用交叉熵，accuracy为正确率
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

'''
4 数据预处理
原来的训练集为（60000，28，28），一共有60000个元素，每个元素是一个二维数组，现在将其转化为一维数组
将灰度图使用astype('float32')/255的方式转换成0-1之间的浮点数
'''
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

'''
将标签也进行一个格式的更改，方便比对，本例我们使用one-hot方法：
由于图片分为10类，所以将原来数字0-9更改为含有10个元素一维数组形式，例如
7------->[0,0,0,0,0,0,0,1,0,0]
'''
print("before change: ", train_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", train_labels[0])

'''
5 训练
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs: 计算所有60000图片的次数
'''
network.fit(train_images, train_labels, batch_size=128, epochs=5)

'''
6 测试数据输入，检验网络学习后的图片识别效果。
识别效果与硬件有关（CPU/GPU）。
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss', test_loss)
print('test_acc', test_acc)

'''
7 手动输入一张图片测试
'''
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)
print(res)
for i in range(res[1].shape[0]):
    if res[1][i] == 1:
        print(f'the answer is {i}')
        break
'''
test = plt.imread('./dataset/my_own_3.png')
gray = test.mean(axis=-1)
gray = gray.reshape((1, 28*28))
res = network.predict(gray)
print(res)
for i in range(res[0].shape[0]):
    if res[0][i] == 1:
        print(f'the answer is {i}')
        break