import numpy as np
import matplotlib.pyplot as plt

# batch_size = [15, 30, 60, 120, 250, 500]
# acc1 = [0.9931, 0.9943, 0.9926, 0.9948, 0.9943, 0.9944]
# acc2 = [0.9939, 0.9943, 0.9942, 0.9947, 0.9948, 0.9947]
# acc3 = [0.9944, 0.9945, 0.9943, 0.9957, 0.9945, 0.9957]
# acc4 = [0.9937, 0.9954, 0.9940, 0.9954, 0.9957, 0.9968]
# # y1, y2 = np.sin(x), np.cos(x)
# plt.figure(figsize=(10, 5))
#
# # plt.plot(x, y1)
# # plt.plot(x, y2)
# plt.plot(batch_size, acc1, label = 'Data enhancement = 0 and Epochs = 20')
# plt.plot(batch_size, acc2, label = 'Data enhancement = 0 and Epochs = 50')
# plt.plot(batch_size, acc3, label = 'Data enhancement = 1 and Epochs = 20')
# plt.plot(batch_size, acc4, label = 'Data enhancement = 1 and Epochs = 50')
# plt.xticks([15, 30, 60, 120, 250, 500])
# plt.legend(loc='lower right')
# plt.title('The influence of different batch size, data enhancement and epochs on the accuracy of test dataset')
# plt.xlabel('batch_size')
# plt.ylabel('the accuracy of test dataset')
# plt.show()

name = ["1","2","3","4","5"]
# y1 = [6, 5, 8, 5, 6, 6, 8, 9, 8, 10]
y2 = [0.9763, 0.9837, 0.9879, 0.9868, 0.9968]
# y3 = [4, 1, 2, 1, 2, 1, 6, 2, 3, 2]

x = np.arange(len(name))
width = 0.3

# plt.bar(x, y1,  width=width, label='label1',color='darkorange')
plt.bar(x + width, y2, width=width, label='accuracy', color='deepskyblue', tick_label=name)
# plt.bar(x + 2 * width, y3, width=width, label='label3', color='green')

# 显示在图形上的值
# for a, b in zip(x,y1):
#     plt.text(a, b+0.1, b, ha='center', va='bottom')
for a,b in zip(x,y2):
    plt.text(a+width, b+0.001, b, ha='center', va='bottom')
# for a,b in zip(x, y3):
#     plt.text(a+2*width, b+0.1, b, ha='center', va='bottom')

plt.xticks()
plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('Accuracy')
plt.xlabel('Experiment Number')
plt.ylim(0.97, 1.0, 0.02)
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
plt.title("Highest accuracy in each experiment")
# plt.savefig('D:\\result.png')
plt.show()