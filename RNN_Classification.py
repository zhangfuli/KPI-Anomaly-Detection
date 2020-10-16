from torch import nn
import torch


class RNN_Classification(nn.Module):
    def __init__(self):
        super(RNN_Classification, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,  # 时间窗口的数量
            hidden_size=64,
            num_layers=1,   # 层数
            batch_first=True,  # (batch, timestamp, input)
        )
        self.out = nn.Linear(64, 10)  # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :]) # (batch, timestamp, input)
        return out

