# 实验4

import os
import tensorflow as tf

class CNN(object):
    def __init__(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model

class DataSource(object):
    def __init__(self):
        # mnist数据集存储的位置，如果不存在将自动下载
        data_path = os.path.abspath(os.path.dirname(__file__))
        # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=data_path)
        import input_data
        mnist_data = input_data.read_data_sets('mnist_data', one_hot=False)

        # 训练数据
        train_images = mnist_data.train.images
        train_labels = mnist_data.train.labels

        # 测试数据
        test_images = mnist_data.test.images
        test_labels = mnist_data.test.labels


        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((-1, 28, 28, 1))
        test_images = test_images.reshape((-1, 28, 28, 1))
        # 像素映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        check_path = 'best_model/mnist_{epoch:02d}-{val_accuracy:.2f}.hdf5'
        checkpointer = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        if not os.path.exists('model_3.h5'):
            history = self.cnn.model.fit(self.data.train_images, self.data.train_labels, epochs=20, callbacks=[checkpointer])
            self.cnn.model.save('model_3.h5')

            from matplotlib.ticker import MultipleLocator
            import matplotlib.pyplot as plt
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
            ax.set_xlim(0, 29)
            ax.set_ylim(0.90, 1.0)
            ax.yaxis.set_major_locator(y_major_locator)
            ax2.set_ylabel("Loss")
            # ax2.legend(loc=0)
            plt.title("Model Accuracy and Loss")
            plt.show()
            plt.savefig('model_3.png')

        else:
            self.cnn.model.load_weights('model_3.h5')

        test_loss, test_acc = self.cnn.model.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))

if __name__=="__main__":
    app = Train()
    app.train()