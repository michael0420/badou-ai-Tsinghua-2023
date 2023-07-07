import numpy as np

class NeuralNetwork:
    pass

'''
    1、读取数据
'''
with open('../dataset/mnist_test.csv') as f:
    data_list = f.readlines()

# 截取属于第一幅图像的数据
values = data_list[0].split(',')
img_array = np.asfarray(values[1:]).reshape((28, 28))
# 数据归一化
img_array = img_array / 255 * 0.99 + 0.01
# 初始化这张图片的标签，one-hot
# 输出层有十个节点
targets = np.zeros(10) + 0.01
targets[int(values[0])] = 0.99
print(targets) 

'''
    2、初始化神经网络
'''
# 由于输入图像为28*28，所以输入节点有28*28个
inodes = 28*28
# 中间层的节点我们选择了100个神经元，这个选择是经验值。
# 中间层的节点数没有专门的办法去规定，其数量会根据不同的问题而变化。
# 确定中间层神经元节点数最好的办法是实验，不停的选取各种数量，看看那种数量能使得网络的表现最好。
hnodes = 100
onodes = 10
lrate = 0.3

n = NeuralNetwork(inodes, hnodes, onodes, lrate)

'''
    3、根据第一步读入训练数据
'''
with open('../dataset/mnist_train.csv') as f:
    train_data_list = f.readlines()

for record in train_data_list:
    values = record.split(',')
    img_values = values[1:]
    inputs = np.asfarray(img_values) / 255 * 0.99 + 0.01
    # 设置对应关系
    targets = np.zeros(onodes) + 0.01
    targets[int(values[0])] = 0.99
    # 训练
    n.train(inputs, targets)

'''
    3、最后将所有测试图片输入，查看检测效果
'''
with open('../dataset/mnist_test.csv') as f:
    test_data_list = f.readlines()

scores = []
for record in test_data_list:
    values = record.split(',')
    print("该图片数字是 ", int(values[0]))
    inputs = np.asfarray(values[1:]) / 255 * 0.99 + 0.01
    # 推理
    res = n.query(inputs)
    # 通过argmax找到最大输出的下标
    label = np.argmax(res)
    print("网络认为该图片数字是 ", label)
    if label == int(values[0]):
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算成功率
nd_scores = np.asarray(scores)
print("成功率：", nd_scores.sum() / nd_scores.size)

'''
    4、epoch
    在原来网络训练的基础上再加上一层外循环
    但是对于普通电脑而言执行的时间会很长。
    epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，网络就会引发一个过拟合的问题.
'''
#加入epocs,设定网络的训练循环次数
epochs = 10

for e in epochs:
    for record in train_data_list:
        values = record.split(',')
        img_values = values[1:]
        inputs = np.asfarray(img_values) / 255 * 0.99 + 0.01
        # 设置对应关系
        targets = np.zeros(onodes) + 0.01
        targets[int(values[0])] = 0.99
        # 训练
        n.train(inputs, targets)