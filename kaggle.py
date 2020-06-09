# kaggle
import csv

import tensorflow as tf
import numpy as np
from matplotlib.ticker import MultipleLocator

import input_data
# import keras
# import os

import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

import matplotlib.pyplot as plt

def loadTrainData():
    l = []
    with open('./digit-recognizer/train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    l = np.array(l)
    label = l[:, 0]
    data = l[:, 1:]
    return nomalizing(toInt(data)), toInt(label)

def toInt(array):
    array = np.mat(array)
    m, n = np.shape(array)
    newArray = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray

def nomalizing(array):
    m, n = np.shape(array)
    for i in range(m):
        for j in range(n):
            array[i, j] /= 255
    return array

def loadTestData():
    l = []
    with open('./digit-recognizer/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = np.array(l)
    return nomalizing(toInt(data))

def saveResult(result):
    with open('result.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile)
        # for i in result:
        #     tmp = []
        #     tmp.append(i)
        #     myWriter.writerow(tmp)
        for i in result:
            myWriter.writerow(i)

train_images, train_labels = loadTrainData()
test_images = loadTestData()

train_images = train_images.reshape((-1, 28, 28, 1))
train_labels = keras.utils.to_categorical(train_labels, 10)[0]


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
# keras.utils.plot_model(sqeue, show_layer_names=True, to_file='model_plot_6.png')
sqeue.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# 数据增强
data_augment = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # 在 [0, 指定角度] 范围内随机角度旋转
    zoom_range=0.1,     # 当制定一个数时，图片同时在长宽两个方向进行同等程度的放缩操作
    width_shift_range=0.1,  # 水平位置平移
    height_shift_range=0.1, # 上下位置平移
)


model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='best_model/mnist_{epoch:02d}-{val_accuracy:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=True)
print('Fitting model...')

if not os.path.exists('model_9.h5'):
    history = sqeue.fit_generator(data_augment.flow(train_images, train_labels, batch_size=500, shuffle=True), steps_per_epoch = train_images.shape[0] // 500, epochs=50, verbose=1, callbacks=[model_checkpoint])


    # from matplotlib.ticker import MultipleLocator
    # # 绘制训练 & 验证的准确率值
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # lns1 = ax.plot(history.history['acc'], '-', label='Train accuracy')
    # # plt.plot(history.history['val_accuracy'])
    # ax2 = ax.twinx()
    # lns2 = ax2.plot(history.history['loss'], '-r', label='Train loss')
    # lns = lns1 + lns2
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc=0)
    # # ax.legend(loc=0)
    # ax.grid()
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Accuracy")
    # x_major_locator = MultipleLocator(5)
    # y_major_locator = MultipleLocator(0.004)
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.set_xlim(0, 50)
    # ax.set_ylim(0.96, 1.0)
    # ax.yaxis.set_major_locator(y_major_locator)
    # ax2.set_ylabel("Loss")
    # # ax2.legend(loc=0)
    # plt.title("Model Accuracy and Loss")
    # plt.show()
    # plt.savefig('model_8.png')

    sqeue.save('model_9.h5')

else:
    sqeue.load_weights('model_9.h5')

# score = sqeue.evaluate(x=test_images, y=test_labels)

# 输出 loss 和 accuracy
# print("loss",score[0])
# print("accu",score[1])


result = []

for i in range(len(test_images)):
    pred = sqeue.predict(test_images[i].reshape(-1, 28, 28, 1))
    result.append(np.argmax(pred))
# saveResult(result)

import pandas as pd

# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'Label': result})

# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("result.csv", index=False, sep=',')