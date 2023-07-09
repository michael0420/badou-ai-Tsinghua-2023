from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np

#加载数据
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

# digit = test_images[0]
# plt.imshow(digit,cmap=plt.cm.binary)
# plt.show()

#搭建网络
network = models.Sequential()
network.add(layers.Dense(units=512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#归一化
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float64') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float64') / 255

#onehot
train_labels = to_categorical(train_labels)
#print(test_labels[0])
test_labels = to_categorical(test_labels)
#print(test_labels[0])

#训练
network.fit(train_images,train_labels,128,5,verbose=0)
network.summary()
#测试
test_loss,test_acc = network.hidden_inputs(test_images,test_labels)
print(test_loss)
print('test_acc',test_acc)

res = network.predict(test_images)
print(test_labels[1])
print(np.argmax(res[1]))
