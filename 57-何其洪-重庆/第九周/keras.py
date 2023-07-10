# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical


if __name__ == '__main__':
    # 加载数据集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Sequential()方法是一个容器，描述了神经网络的网络结构
    network = models.Sequential()
    # 全连接层 layers.Dense(神经元个数，activation = "激活函数“，input_shape=输入数据格式）
    # 其中：activation可选 relu 、softmax、 sigmoid、 tanh等
    # input_shape=(28*28,)表示当前处理层接收的数据格式必须是长和宽都是28的二维数组
    network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    network.add(layers.Dense(10, activation='softmax'))
    """ 
    model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
    model.compile(optimizer = 优化器，loss = 损失函数，metrics = ["准确率”])
    optimizer可以是字符串形式给出的优化器名字，也可以是函数形式，使用函数形式可以设置学习率、动量和超参数
    例如："sgd" 或者 tf.optimizers.SGD(lr = 学习率，decay = 学习率衰减率，momentum = 动量参数）
         "adagrad" 或者 tf.keras.optimizers.Adagrad(lr = 学习率，decay = 学习率衰减率）
         "adadelta"  或者  tf.keras.optimizers.Adadelta(lr = 学习率，decay = 学习率衰减率）
         "adam"  或者  tf.keras.optimizers.Adam(lr = 学习率，decay = 学习率衰减率)
    
    loss可以是字符串形式给出的损失函数的名字，也可以是函数形式
    例如："mse" 或者 tf.keras.losses.MeanSquaredError()
         "sparse_categorical_crossentropy"  或者  tf.keras.losses.SparseCatagoricalCrossentropy(from_logits = False)
         损失函数经常需要使用softmax函数来将输出转化为概率分布的形式，在这里from_logits代表是否将输出转为概率分布的形式，
         为False时表示转换为概率分布，为True时表示不转换，直接输出
    
    metrics标注网络评价指标
    例如："accuracy": y_ 和 y 都是数值，如y_ = [1] y = [1]  #y_为真实值，y为预测值
         "sparse_accuracy": y_和y都是以独热码 和概率分布表示，如y_ = [0, 1, 0], y = [0.256, 0.695, 0.048]
         "sparse_categorical_accuracy": y_是以数值形式给出，y是以 独热码给出，如y_ = [1], y = [0.256 0.695, 0.048]
    """
    network.compile(optimizer=optimizers.RMSprop(), loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])
    """
    将数据做归一化处理
    1. train_images数组长度为60000，每个元素是一个28行28列的二维数组，将二维数组转化为28*28个元素的一维数组
    2. 训练集的图片为灰度图，train_images.astype('float32') / 255 可将每个像素点的值转变为[0, 1]之间的浮点数
    """
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    # 将标签转换为：one hot（独热编码）形式 [0,0,2,0,0,0,0,0,0,0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # network.fit(x=训练数据集, y=训练标签, batch_size=每批训练大小, epochs=迭代次数)
    network.fit(train_images, train_labels, 128, 10)
    # 使用测试数据集检验训练结果
    testLoss, testAcc = network.evaluate(test_images, test_labels)
    print('成功率：', testAcc)
