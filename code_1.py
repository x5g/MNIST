# 实验1

# 采用全连接的方式训练 手写数字MNIST数据集
import tensorflow as tf
import numpy as np
from matplotlib.ticker import MultipleLocator

import input_data
import matplotlib.pyplot as plt
import os

# 参数是数据所在的路径，如果文件不存在则会自动下载
# MNIST数据集中包含55000个训练数据和10000个数据
# 大小为28×28的手写数字图片 但存储方式为784维向量
mnist_data = input_data.read_data_sets('mnist_data', one_hot=False)

# 训练数据
train_images = mnist_data.train.images
train_labels = mnist_data.train.labels

# 测试数据
test_images = mnist_data.test.images
test_labels = mnist_data.test.labels

# 模型查找 如果不存在则开始训练
if not os.path.exists('model_1.h5'):
    # 模型的输入 输入维度784
    input_ = tf.keras.Input(shape=(784, ))
    # 第一层 全连接层 units = 128, 表示该层有128个神经元 activation激活函数，这里选择tanh 参数总量为784×128+偏置128=100480
    dense1 = tf.keras.layers.Dense(128, activation='relu')(input_)
    # 输出层 全连接层
    # 不让神经网络输出一个标量，代表预测的数字，而是让神经网络分别预测出10个数，分别代表神经网络认为图片是哪个数字的概率，概率越高可能性越大。
    # 这样的话，我们网络的output_dim就应该等于10了，10表示有10个种类
    out = tf.keras.layers.Dense(10, activation='softmax')(dense1)
    # 这一步等价于
    # dense_layer = tf.keras.layer.Dense(10)
    # out = dense_layer(dense2)

    # 绑定形成模型
    model = tf.keras.Model(inputs=input_, outputs=out)
    # 查看模型信息
    model.summary()
    # 编译模型，这一步操作是训练之前必须进行的，目的是绑定优化器，损失函数还有一些指标
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    # 一个模型如果编译之后，就可以进行训练，keras提供了Model.fit(x, y, epochs, ...)，其中x是模型的输入，y是正确的标签，epochs是训练的轮数。
    # 训练的原理大致是模型根据输出得到预测，优化器根据模型预测值与真实值的差距，反向的惩罚模型，使模型更趋向于正确的预测。
    history = model.fit(x=train_images, y=train_labels, epochs=20)

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
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(0.01)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlim(0, 19)
    ax.set_ylim(0.90, 1.0)
    ax.yaxis.set_major_locator(y_major_locator)
    ax2.set_ylabel("Loss")
    # ax2.legend(loc=0)
    plt.title("Model Accuracy and Loss")
    plt.show()
    plt.savefig('model_1.png')
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()


    # 训练完后保存模型 也可保存权重save_weights
    model.save('model_1.h5')
else:
    model = tf.keras.models.load_model('model_1.h5')

# 在测试集中评估
model.evaluate(x=test_images, y=test_labels)

# 图片展示
# for i in range(10):
#     pred = model(tf.expand_dims(test_images[i], axis=0))
#     img = np.reshape(test_images[i], (28, 28))
#     lab = test_labels[i]
#     print('真实标签: ', lab, '， 网络预测: ', np.argmax(pred.numpy()))
#     plt.imshow(img)
#     plt.show()
