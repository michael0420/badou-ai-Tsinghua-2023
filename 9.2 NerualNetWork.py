
import numpy as np
import scipy.special 

class NerualNetWork:
  
    def __init__(self,inputnodes,hiddennodes,outnodes,learningrate):
        #初始化神经网络,输入层，隐藏层，输出层
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outnodes
        
        #设置学习率
        self.lr = learningrate

        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        wih生成为一个服从均值0.0、标准差pow(self.hnodes,-0.5)、形状(self.hnodes,self.inodes)的矩阵，形状要符合与输入矩阵相乘
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        
        '''
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        '''
        每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        '''
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self,input_list,targets_list):
        #根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        np.array是NumPy库中的一个函数，用于将其他数组对象转换为一个NumPy数组。
        object：需要转换为NumPy数组的对象，可以是列表、元组、集合、数组等。
        dtype：指定转换后的数组的数据类型，默认为None，即自动根据对象确定数据类型。
        copy：如果为True，则返回一个新的副本，与原对象无关；如果为False，则返回原始对象的引用，不创建新的副本。
        order：指定数组的存储顺序，默认为'K'，表示按照原始对象的存储顺序进行存储。
        subok：如果为True，则允许转换包含多个子对象的对象（如列表、元组等），默认为False。
        ndmin：指定返回的数组的最小维度数，默认为0，即表示返回的数组至少为0维。
        '''
        inputs = np.array(input_list,ndmin =2).T
        targets = np.array(targets_list,ndmin=2).T
         #计算隐藏层的输入数据
        hidden_inputs = np.dot(self.wih,inputs)
        #隐藏层输入经过激活函数后的输出
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算输出层的输入数据
        final_inputs = np.dot(self.who,hidden_outputs)
        #计算输出层输出数据
        final_outputs = self.activation_function(final_inputs)

        #计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors*final_outputs*(1-final_outputs))
        #根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr*np.dot(output_errors*final_outputs*(1-final_outputs),np.transpose(hidden_outputs))
        self.wih += self.lr*np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),np.transpose(inputs))
        pass


    def query(self,inputs):
        #推理输入的数据
        #计算隐藏层的输入数据
        hidden_inputs = np.dot(self.wih,inputs)
        #隐藏层输入经过激活函数后的输出
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算输出层的输入数据
        final_inputs = np.dot(self.who,hidden_outputs)
        #计算输出层输出数据
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


#初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 200
out_nodes =10
learningrate = 0.1
n = NerualNetWork(inputnodes=input_nodes,hiddennodes=hidden_nodes,outnodes=out_nodes,learningrate=learningrate)


#读入训练数据
#open函数里的路径根据数据存储的路径来设定
train_data_file =  open("mnist_train.csv",'r')  
train_data_list = train_data_file.readlines()
train_data_file.close()

#加入epocs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    for record in train_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0*0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(out_nodes) + 0.1
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)

test_data_file = open("mnist_test.csv",'r') 
test_data_list = test_data_file.readlines()
test_data_file.close()
score = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print('实际数字是：',correct_number)
    inputs = (np.asfarray(all_values[1:]))/255.0*0.99 + 0.01
    #让网络判断图片对应的数字
    outputs = n.query(inputs)
    #找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if correct_number == label :
        score.append(1)
    else:
        score.append(0)
print(score)
#计算图片判断的成功率
score_array = np.asfarray(score)
print("perfermance = ", score_array.sum() / score_array.size)


