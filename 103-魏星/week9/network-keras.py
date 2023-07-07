from keras.datasets import mnist
import keras.models as models
import keras.layers as layers
import keras.utils as utils
import cv2

# 读取训练数据、测试数据 训练数据为28*28的图片
(x_train, y_train), (x_test, y_test) = mnist.load_data("mnist.npz")
print('train_images.shape=',x_train.shape)
print('train_labels.shape=',y_train.shape)
print('test_images.shape=',x_test.shape)
print('test_labels.shape=',y_test.shape)

# 构建模型
network = models.Sequential()
network.add(layers.Dense(units=512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(units=10, activation='softmax'))

# 设置损失函数
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = x_train.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
train_labels = utils.to_categorical(y_train)

test_images = x_test.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
test_labels = utils.to_categorical(y_test)

# 训练
network.fit(train_images, train_labels, batch_size=128, epochs=10)

# 测试
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)

# # 验证
# test_img = cv2.imread("test1.png")
# img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
# img = img.reshape((1, 28*28))
# result = network.predict(img)
# for i in range(result.shape[1]):
#     if (result[0][i] == 1):
#         print("the number for the picture is : ", i)
#         break

result = network.predict(test_images)
print(result.shape)
# for i in range(result.shape[0]):
for j in range(result.shape[1]):
    if result[1][j] == 1:
        print("the number for the picture is : ", j)
        cv2.imwrite("test_"+str(j)+".png",x_test[1])
        break
