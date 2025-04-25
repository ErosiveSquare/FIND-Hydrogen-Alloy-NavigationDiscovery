#用来训练从133维度到化学式向量
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transform import *
from get_target import *
import os
import torch.nn as nn


# 初始化标准化器
file_path = "Composition control absorption1_magpie2.xlsx"

# 训练数据
train_data = GET_X(file_path)
test_data = get_target(file_path)

class My_model4(nn.Module):
    def __init__(self):
        super(My_model4, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,133),
        )
    def forward(self, x):
        return self.model(x)



#开模型
model = My_model4()

#损失函数
criterion = nn.MSELoss()

# 定义L1正则化系数
lambda_l1 = 0.00001
#优化器
learing_rate = 0.001
#梯度裁剪
clip_value = 0.0001
#读取保存的权重
file_name ="weight_data.pth"
# state_dict = torch.load(file_name)
# model.load_state_dict(state_dict)

#梯度下降函数
# optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

# #训练次数
# epochs1 = 10

# sumloss = 0
# for epoch in range(epochs1):
#     for index in range(len(train_data)):
#         input = train_data[index]
#         output = model(input)
#         targets = test_data[index]
#         loss = criterion(output,targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(epoch)

# torch.save(model.state_dict(), 'magpie_.pth')

# input = train_data[0]
# output = model(input)
# print(test_data[0])
# print(output)