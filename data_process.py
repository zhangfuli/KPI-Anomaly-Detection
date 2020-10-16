import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from torch.utils.data import Dataset, DataLoader
import torch

# matplotlib.use('Qt5Agg')
from torch.utils.data import DataLoader

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('./yahoo/A1Benchmark/real_1.csv')
anomaly = data.loc[data['is_anomaly'] == 1]

# plt.plot(data['timestamp'], data['value'], 'b-')
# plt.plot(anomaly['timestamp'], anomaly['value'], 'r*')
# plt.savefig('./images/test2.jpg')
# plt.clf()
# plt.show()
'''
 滑动窗口整理数据
 size: 窗口大小, 序列长度
 step: 步长
 padding: 左右填充大小, 以首位末位数据填充(合不合理？)
 '''
def sliding(data, labels, size, step=1, padding=0, scale=0.2):
    data = data.to_numpy()

    head_data = [data[0] for i in range(padding)]
    tail_data = [data[len(data) - 1] for i in range(padding)]

    head_label = [labels[0] for i in range(padding)]
    tail_label = [labels[len(labels) - 1] for i in range(padding)]
    X = np.concatenate([head_data, data, tail_data])
    y = np.concatenate([head_label, labels, tail_label])

    windows_data = []
    windows_label = []
    if len(X) != len(y):
        print("The data length is inconsistent! len(X) != len(y)")
        return
    for window_index in range(0, len(X) - size, step):
        window_data = X[window_index: window_index + size]
        window_label = 0
        for label_index in range(window_index, window_index + size):
            if y[label_index] == 1:
                window_label = 1
        windows_data.append(window_data)
        windows_label.append(window_label)
    windows_data = torch.tensor(windows_data)
    windows_label = torch.tensor(windows_label)

    positive = windows_data[torch.where(windows_label == 1)]
    negative = windows_data[torch.where(windows_label == 0)]
    print("正样本数(异常点): " + str(positive.size()))
    print("负样本数: " + str(negative.size()))

    # negative = negative[:20]
    # positive = negative[:20]

    positive_train_x = positive[int(positive.size(0) * scale):]
    positive_train_y = torch.tensor([1 for i in range(positive_train_x.size(0))])
    positive_test_x = positive[:int(positive.size(0) * scale)]
    positive_test_y = torch.tensor([1 for i in range(positive_test_x.size(0))])

    negative_train_x = negative[int(negative.size(0) * scale):]
    negative_train_y = torch.tensor([0 for i in range(negative_train_x.size(0))])
    negative_test_x = negative[:int(negative.size(0) * scale)]
    negative_test_y = torch.tensor([0 for i in range(negative_test_x.size(0))])

    train_windows_data = torch.cat([positive_train_x, negative_train_x], dim=0)
    train_windows_label = torch.cat([positive_train_y, negative_train_y], dim=0)
    test_windows_data = torch.cat([positive_test_x, negative_test_x], dim=0)
    test_windows_label = torch.cat([positive_test_y, negative_test_y], dim=0)
    return train_windows_data, train_windows_label, test_windows_data, test_windows_label


# train_windows_data, train_windows_label, test_windows_data, test_windows_label = \
#     sliding(data['value'], data['is_anomaly'], size=400, step=1, padding=0, scale=0.2)


# train_windows_data, train_windows_label, test_windows_data, test_windows_label = \
#     sliding(data['value'], data['is_anomaly'], size=400, step=1, padding=0, scale=0.2)
# print(train_windows_data.size(), train_windows_label.size(), test_windows_data.size(), test_windows_label.size())


# class SequenceTrainDataSet(Dataset):
#     def __init__(self, train_windows_data, train_windows_label):
#         # self.path = path
#         # self.data = pd.read_csv(self.path)
#         self.train_windows_data = train_windows_data
#         self.train_windows_label = train_windows_label
#
#     def __getitem__(self, index):
#         return self.train_windows_data[index], self.train_windows_label[index]
#
#     def __len__(self):
#         return self.train_windows_data.size(0)
#
#
# class SequenceTestDataSet(Dataset):
#     def __init__(self, path, size=400, step=1, padding=0, scale=0.2):
#         self.path = path
#         self.data = pd.read_csv(self.path)
#         _, _, self.test_windows_data, self.test_windows_label = sliding(self.data['value'], self.data['is_anomaly'],
#                                                                         size=size,
#                                                                         step=step, padding=padding, scale=scale)
#
#     def __getitem__(self, index):
#         return self.test_windows_data[index], self.test_windows_label[index]
#
#     def __len__(self):
#         return self.test_windows_data.size(0)
