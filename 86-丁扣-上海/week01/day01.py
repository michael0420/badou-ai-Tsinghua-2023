# # %matplotlib inline
# import torch
# from IPython import display
# from matplotlib import pyplot as plt
# import numpy as np
# import random
# from time import time
# from torch import nn
# import torch.utils.data as Data
#
# num_inputs = 2  # 特征
# num_examples = 1000  # 数据集
# true_w = [2, -3.4]
# true_b = 4.2
# features = torch.randn(num_examples, num_inputs,
#                        dtype=torch.float32)
# # print(features)
# # print(features.shape)
# # print(features.size)
#
# labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# # print(f'--labels: {labels}')
# # print(features[0], labels[0])
# ztfb = torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)  # 0~0.01的正态分布，size是等于标签的数据大小
#
# labels += ztfb
# # print(labels)
#
#
# # def use_svg_display():
# #     # 用矢量图显示
# #     display.set_matplotlib_formats('svg')
# #
# #
# # def set_figsize(figsize=(3.5, 2.5)):
# #     use_svg_display()
# #     # 设置图的尺寸
# #     plt.rcParams['figure.figsize'] = figsize
#
# # # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# # import sys
# # sys.path.append("..")
# # from d2lzh_pytorch import *
#
# # print(features[:, 1].numpy())
# # print("&&" * 30)
# # print(labels.numpy())
#
# # set_figsize()
# # plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# # plt.show()
#
#
# # print(labels)
# batch_size = 10
# # 将训练数据的特征和标签组合
# dataset = Data.TensorDataset(features, labels)
# # 随机读取小批量
# data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# for X, y in data_iter:
#     print(X)
#     print(y)
#     break
#
#
# class LinearNet(nn.Module):
#     def __init__(self, n_feature):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(n_feature, 1)
#
#     # forward 定义前向传播
#     def forward(self, x):
#         y = self.linear(x)
#         return y
#
#
# net = LinearNet(num_inputs)
# print(net)  # 使用print可以打印出网络的结构
#
#
