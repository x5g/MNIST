# 实验3

# 输入图片尺寸为[batch_size, old_w, old_h, old_c]，若卷积层中卷积核的个数为new_c个，卷积核的尺寸为size，步长为stride
# 填充为valid的话，则输出尺寸应该是[batch_size, new_w, new_h, new_c]，其中new_w = (old_w - size) / stride + 1, new_h = (new_h - size) / stride + 1
# 若padding是same，则new_w = old_w // stride, new_h = old_h // stride，整除步长、向上取正
import tensorflow as tf

# a = tf.random.normal(shape=[1, 7, 7, 3]) # 假设这是一张7*7的3通道图片
#
# conv1 = tf.keras.layers.Conv2D(5, 3, 2, 'valid')(a) # 5是卷积核的个数，3是卷积核的尺寸，2是步长
# conv2 = tf.keras.layers.Conv2D(5, 3, 2, 'same')(a)
#
# print(conv1.shape)  # 输出(1, 3, 3, 5) 1是批大小，5是通道数，也就是有5个特征
# print(conv2.shape)

import input_data
import os
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

dataset = input_data.read_data_sets('mnist_data', one_hot=True)

train_dataset = dataset.train
test_dataset = dataset.test

if not os.path.exists('model_2.h5'):
    input_ = tf.keras.Input(shape=(784, ))
    input_reshape = tf.reshape(input_, (-1, 28, 28, 1)) # -1表示批次维度，只有进行训练时才知道是几。一定要包括深度维

    conv1 = tf.keras.layers.Conv2D(16, 3, 2, activation='relu')(input_reshape)
    conv2 = tf.keras.layers.Conv2D(32, 3, 2, activation='relu')(conv1)
    conv3 = tf.keras.layers.Conv2D(64, 3, 2, activation='relu')(conv2)

    flatten = tf.keras.layers.Flatten()(conv3)  # 因为全连接的输入是向量，所以要把特征展开成向量

    fc1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    result = tf.keras.layers.Dense(10, activation='softmax')(fc1)

    model = tf.keras.Model(input_, result)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    history = model.fit(train_dataset.images, train_dataset.labels, batch_size=32, epochs=50)

    model.save('model_2.h5')

    # 绘制训练 & 验证的准确率值
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(history.history['accuracy'], '-', label='Train accuracy')
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
    y_major_locator = MultipleLocator(0.01)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlim(0, 49)
    ax.set_ylim(0.93, 1.0)
    ax.yaxis.set_major_locator(y_major_locator)
    ax2.yaxis.set_major_locator(MultipleLocator(0.02))
    ax2.set_ylabel("Loss")
    ax2.set_ylim(0, 0.25)
    # ax2.legend(loc=0)
    plt.title("Model Accuracy and Loss")
    plt.show()
    plt.savefig('model_2.png')

else:
    model = tf.keras.models.load_model('model_2.h5')

model.evaluate(test_dataset.images, test_dataset.labels)
