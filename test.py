import torch
import torch.utils.data as Data

x = torch.randn(3, 3, 3)  # 随机生成一些数据
print(x)
print(x.shape)
print(x[:, :, -1])
