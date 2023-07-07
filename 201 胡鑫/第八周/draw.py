import matplotlib.pyplot as plt
import numpy as np

file = open('../dataset/mnist_test.csv')
data_list = file.readlines()
file.close()

print(len(data_list))
print(data_list[0])

# 将第一张图片数据分离出来
img_values = data_list[0].split(',')

img = np.asfarray(img_values[1:]).reshape((28, 28))
# print(img)
# Greys为灰度图，gray只有黑白
plt.imshow(img, cmap='Greys')
plt.show()

# 数据归一化
scaled_input = img / 255 * 0.99 + 0.01
print(scaled_input)