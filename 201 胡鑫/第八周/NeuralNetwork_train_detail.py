import numpy as np

'''
训练函数的实现：
    1、训练函数前面的步骤与推理的步骤完全相同，此次训练的数据传入的是一个横向量
    需要先将其转换为对应形式方便计算
    inputs_list：训练的数据
    targets_list：训练数据对应的正确结果
'''
def train(self, inputs_list, targets_list):
    # 为后面的矩阵计算做准备，ndmin=2，表示创建的是一个二维的ndarray型，.T表示做转置转换成列向量
    inputs = np.array(inputs_list, ndmin=2).T
    targets = np.array(targets_list, ndmin=2).T
    # 首先计算出隐藏层的输入信号
    hidden_inputs = np.dot(self.wih, inputs)
    # 通过激活函数计算出隐藏层的输出信号
    hidden_outputs = self.activation_function(hidden_inputs)
    # 再计算输出层的输入信号
    final_inputs = np.dot(self.who, hidden_outputs)
    # 最后的输出信号
    final_outputs = self.activation_function(final_inputs)

'''
    2、在计算出最后的输出后，需要根据正确结果计算误差，然后反向传播更新权重值
'''
def train(self, inputs_list, targets_list):
    # 为后面的矩阵计算做准备，ndmin=2，表示创建的是一个二维的ndarray型，.T表示做转置转换成列向量
    inputs = np.array(inputs_list, ndmin=2).T
    targets = np.array(targets_list, ndmin=2).T
    # 首先计算出隐藏层的输入信号
    hidden_inputs = np.dot(self.wih, inputs)
    # 通过激活函数计算出隐藏层的输出信号
    hidden_outputs = self.activation_function(hidden_inputs)
    # 再计算输出层的输入信号
    final_inputs = np.dot(self.who, hidden_outputs)
    # 最后的输出信号
    final_outputs = self.activation_function(final_inputs)

    # 计算误差
    output_errors = (targets - final_outputs)**2
    # 隐藏层的梯度可以通过下一层（即输出层）的误差、当前层的权重矩阵的转置和当前层的激活函数的导数来计算
    # 这个计算过程实际上利用了链式法则来传播误差，并计算每一层对误差的贡献
    hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
    # 更新权重值
    self.wih += self.lr * np.dot(output_errors * final_outputs *(1 - final_outputs), hidden_outputs.T)
    self.who += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), 
                                 inputs.T)
    
'''
    3、使用数据训练神经网络
'''
data_file = open('../dataset/mnist_test.csv')
data_list = data_file.readlines()
data_file.close()
# 用draw.py绘制出data_list[0]

'''
    4、通过draw.py我们看出数据代表的是一个灰度图，我们将数据预处理（归一化）一下，
    将所有的值换算到0.01至1.0之间。
    由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，
    就能让数据全部落入到0和1之间。
    有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题。
    所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
'''
scaled_input = np.asfarray(data_list[0].split(',')[1:]) / 255 * 0.99 + 0.01
