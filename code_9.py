# 实验5 1-6

import tensorflow as tf
import numpy as np
from matplotlib.ticker import MultipleLocator

import input_data
import keras
import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import matplotlib.pyplot as plt
import os

# # 参数是数据所在的路径，如果文件不存在则会自动下载
# # MNIST数据集中包含55000个训练数据和10000个数据
# # 大小为28×28的手写数字图片 但存储方式为784维向量
# mnist_data = input_data.read_data_sets('mnist_data', one_hot=True)
#
# # 训练数据
# train_images = mnist_data.train.images
# train_labels = mnist_data.train.labels
#
# # 测试数据
# test_images = mnist_data.test.images
# test_labels = mnist_data.test.labels
#
# # 6万张训练图片，1万张测试图片
# train_images = train_images.reshape((-1, 28, 28, 1))
# test_images = test_images.reshape((-1, 28, 28, 1))
# # 像素映射到 0 - 1 之间
# train_images, test_images = train_images / 255.0, test_images / 255.0

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(60000,784)
# x_test = x_test.reshape(10000,784)
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# 归一化
x_train = x_train/255
x_test = x_test/255

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

train_images = x_train
train_labels = y_train
test_images = x_test
test_labels = y_test

# 构建网络
sqeue = keras.models.Sequential()
# 第一个卷积层，32个卷积核，大小5×5，卷积模式SAME，激活函数relu，输入张量的大小
sqeue.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
sqeue.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
# 池化层，池化核大小2×2
sqeue.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 随机丢弃四分之一的网络连接，防止过拟合
sqeue.add(keras.layers.Dropout(rate=0.25))
sqeue.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
sqeue.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
sqeue.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
sqeue.add(keras.layers.Dropout(rate=0.25))
# 全连接层，展开操作
sqeue.add((keras.layers.Flatten()))
# 添加隐藏层神经元的数量和激活函数
sqeue.add(keras.layers.Dense(256, activation='relu'))
sqeue.add(keras.layers.Dropout(rate=0.25))
# 输出层
sqeue.add(keras.layers.Dense(10, activation='softmax'))
sqeue.summary()
keras.utils.plot_model(sqeue, show_layer_names=True, to_file='model_plot_5.png')
sqeue.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if not os.path.exists('model_5.h5'):
    history = sqeue.fit(x=train_images, y=train_labels, epochs=50, batch_size=50)

    from matplotlib.ticker import MultipleLocator
    # 绘制训练 & 验证的准确率值
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(history.history['acc'], '-', label='Train accuracy')
    # plt.plot(history.history['val_accuracy'])
    ax2 = ax.twinx()
    lns2 = ax2.plot(history.history['loss'], '-r', label='Train loss')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    # ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    x_major_locator = MultipleLocator(2)
    y_major_locator = MultipleLocator(0.004)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlim(0, 20)
    ax.set_ylim(0.96, 1.0)
    ax.yaxis.set_major_locator(y_major_locator)
    ax2.set_ylabel("Loss")
    # ax2.legend(loc=0)
    plt.title("Model Accuracy and Loss")
    plt.show()
    plt.savefig('model_5.png')

    sqeue.save('model_5.h5')

else:
    sqeue.load_weights('model_5.h5')
score = sqeue.evaluate(x=test_images, y=test_labels)

# 输出 loss 和 accuracy
print("loss",score[0])
print("accu",score[1])