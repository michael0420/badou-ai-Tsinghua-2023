from tensorflow.keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images[0].shape)



from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


from tensorflow.keras.utils import to_categorical
print('test_label[0]:{}'.format(test_labels[0]))
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('test_label[0]:{}'.format(test_labels[0]))

network.fit(train_images, train_labels, epochs=1, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss:{}, test_acc:{}'.format(test_loss, test_acc))


digit = test_images[0].reshape((28, 28))
print(digit.shape)
# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
# test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)
print(res[0])
for i in range(100):
    max_idx = 0
    max_val = res[i][0]
    for j in range(10):
        if res[i][j] > max_val:
            max_idx = j
    print(test_labels[i], max_idx)






