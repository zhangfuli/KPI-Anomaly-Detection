from torch.utils.data import TensorDataset
from torch.autograd import Variable
from TCN import TCN
from data_process import *
import torch.optim as optim
import torch
import torch.nn.functional as F
import pandas as pd
from tensorboardX import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter('run/scalar')

size = 400
input_channels = 1
n_classes = 2
batch_size = 32
kernel_size = 5
epochs = 5000
dropout = 0.0
lr = 0.0001

channel_sizes = [30] * 8

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Producing data...")
path = './yahoo/A1Benchmark/real_1.csv'
data = pd.read_csv(path)

train_windows_data, train_windows_label, test_windows_data, test_windows_label = \
    sliding(data['value'], data['is_anomaly'], size=400, step=1, padding=0, scale=0.2)

train_data = TensorDataset(train_windows_data, train_windows_label)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = TensorDataset(test_windows_data, test_windows_label)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
#
# train_windows_data, train_windows_label, test_windows_data, test_windows_label = \
#     sliding(data['value'], data['is_anomaly'], size=400, step=1, padding=0, scale=0.2)

model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
if device == 'gpu':
    model.cuda()

optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)


def train(epoch):
    model.train()
    total_loss = 0
    TP = 1e-9
    TN = 1e-9
    FP = 1e-9
    FN = 1e-9
    for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
        x = torch.tensor(x, dtype=torch.float32)
        x = Variable(torch.unsqueeze(x, 1))
        y = torch.tensor(y, dtype=torch.long)

        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        output = output.max(1, keepdim=True)[1]
        y = y.view_as(output)

        TP += ((y == 1) & (output == 1)).cpu().sum().numpy()
        TN += ((y == 0) & (output == 0)).cpu().sum().numpy()
        FP += ((y == 0) & (output == 1)).cpu().sum().numpy()
        FN += ((y == 1) & (output == 0)).cpu().sum().numpy()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    presicion = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * presicion * recall / (presicion + recall)
    print('Train Epoch: {:2d} '
          'Learning rate: {:.8f} '
          'Average loss: {:.6f} '
          'presicion: {:.4f} '
          'recall: {:.4f} '
          'accuracy: {:.4f} '
          'f1: {:.4f}'.format(epoch, lr, total_loss / len(train_loader), presicion, recall, accuracy, f1))
    writer.add_scalar('Train', total_loss / len(train_loader), epoch)


def test():
    # model.load_state_dict(torch.load('./run/model/tcn_v1.pt'))
    model.eval()
    evaluate_loss = 0
    TP = 1e-9
    FN = 1e-9
    FP = 1e-9
    TN = 1e-9
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = torch.tensor(x, dtype=torch.float32)
            x = Variable(torch.unsqueeze(x, 1))
            y = torch.tensor(y, dtype=torch.long)
            output = model(x)

            test_loss = F.nll_loss(output, y)

            # 返回每一行的预测值
            pred = output.max(1, keepdim=True)[1]
            y = y.view_as(pred)

            # print(pred)
            # print(y.view_as(pred))
            TP += ((y == 1) & (pred == 1)).cpu().sum().numpy()
            TN += ((y == 0) & (pred == 0)).cpu().sum().numpy()
            FP += ((y == 0) & (pred == 1)).cpu().sum().numpy()
            FN += ((y == 1) & (pred == 0)).cpu().sum().numpy()
            evaluate_loss += test_loss

        presicion = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1 = 2 * presicion * recall / (presicion + recall)

        print('Test set: Average loss: {:.6f}'.format(evaluate_loss / len(test_loader)))
        print('Test set: presicion: {:.6f}'.format(presicion))
        print('Test set: recall: {:.6f}'.format(recall))
        print('Test set: accuracy: {:.6f}'.format(accuracy))
        print('Test set: f1: {:.6f}'.format(f1))


for ep in range(1, epochs + 1):
    train(ep)
    test()
    # if ep % 10 == 0:
    #     lr /= 10
#
#torch.save(model.state_dict(), './run/model/tcn_v1.pt')

