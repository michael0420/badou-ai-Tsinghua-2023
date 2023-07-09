from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers

# 加载
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
# 转one_hot
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练
network.fit(train_images, train_labels, batch_size=128, epochs=5)

# 测试结果
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28*28))

res = network.predict(test_images)

for i in range(res[5].shape[0]):
    if res[5][i] == 1:
        print('number: ', i)
        print('right is: ', test_labels[5])
