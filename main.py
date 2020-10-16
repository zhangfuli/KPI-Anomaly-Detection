import torchvision
from torch.utils.data import DataLoader

from RNN_Classification import RNN_Classification
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Hyper Parameters
    EPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
    BATCH_SIZE = 64
    TIME_STEP = 28  # rnn 时间步数 / 图片高度
    INPUT_SIZE = 28  # rnn 每步输入值 / 图片每行像素
    LR = 0.01  # learning rate
    DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 Fasle

    # Mnist 手写数字
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
    )
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

    # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 为了节约时间, 我们只测试前2000个
    test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000] / 255  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:2000]

    # 训练
    rnn_classification = RNN_Classification()
    print(rnn_classification)

    optimizer = torch.optim.Adam(rnn_classification.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):  # gives batch data
            b_x = Variable(x.view(-1, 28, 28))  # reshape x to (batch, time_step, input_size)
            b_y = Variable(y)  # batch y

            output = rnn_classification(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:  # testing
                test_output = rnn_classification(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y.numpy()) / test_y.size(0)
                print('train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)

    test_output = rnn_classification(test_x[:10].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')