# 实验2

import tensorflow as tf
import matplotlib.pyplot as plt
import input_data
import os

mnist_data = input_data.read_data_sets('mnist_data', one_hot=True)

# 训练数据
train_images = mnist_data.train.images
train_labels = mnist_data.train.labels

# 测试数据
test_images = mnist_data.test.images
test_labels = mnist_data.test.labels

input_ = tf.keras.Input(shape=(784,))

fc1 = tf.keras.layers.Dense(512, activation='relu')(input_)
do1 = tf.keras.layers.Dropout(0.2)(fc1)

fc2 = tf.keras.layers.Dense(512, activation='relu')(do1)
do2 = tf.keras.layers.Dropout(0.2)(fc2)

result = tf.keras.layers.Dense(10, activation='softmax')(do2)

model = tf.keras.Model(input_, result)

model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='best_model/mnist_{epoch:02d}-{val_accuracy:.2f}.hdf5', save_weights_only=True, verbose=1)


if not os.path.exists('model_4.h5'):
    hist = model.fit(train_images, train_labels, batch_size=128, epochs=100, validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)
    model.save_weights('model_4.h5')

    from matplotlib.ticker import MultipleLocator
    # 绘制训练 & 验证的准确率值
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(hist.history['accuracy'], color='blue' ,linestyle='-', label='Train accuracy')
    lns2 = ax.plot(hist.history['val_accuracy'], color='orange', linestyle='-', label='Validation accuracy')
    ax2 = ax.twinx()
    lns3 = ax2.plot(hist.history['loss'], color='red', linestyle='-', label='Train loss')
    lns4 = ax2.plot(hist.history['val_loss'], color='green', linestyle='-', label='Validation loss')
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower right')
    # ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    x_major_locator = MultipleLocator(10)
    y_major_locator = MultipleLocator(0.01)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.90, 1.0)
    ax.yaxis.set_major_locator(y_major_locator)
    ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    ax2.set_ylabel("Loss")
    ax2.set_ylim(0, 0.5)
    # ax2.legend(loc=0)
    plt.title("Model Accuracy and Loss")
    plt.show()
    plt.savefig('model_4.png')

else:
    model.load_weights('model_4.png')

model.evaluate(test_images, test_labels)
