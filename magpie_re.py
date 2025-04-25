#用来训练从133维度到化学式向量
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from transform import *
from get_target import *
import os
import torch.nn as nn
from to_formula import *
import joblib
# 初始化标准化器
# file_path = "Composition control absorption1_magpie2.xlsx"
# scaler = StandardScaler()

# # # 训练数据
# test_data = GET_X(file_path)
# train_data = get_target(file_path)
# train_data = scaler.fit_transform(train_data)
# train_data = torch.tensor(train_data).float()
# print(train_data.shape)
# print(test_data.shape)
# joblib.dump(scaler, 'scaler3.joblib')
class My_model3(nn.Module):
    def __init__(self):
        super(My_model3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(132, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,40),
        )
    def forward(self, x):
        return self.model(x)



#开模型
# model = My_model3()

# #损失函数
# criterion = nn.MSELoss()

# 定义L1正则化系数
lambda_l1 = 0.00001
#优化器
learing_rate = 0.001
#梯度裁剪
clip_value = 0.0001
# 读取保存的权重
# file_name ='magpie_reverse.pth'
# state_dict = torch.load(file_name)
# model.load_state_dict(state_dict)

#梯度下降函数
# optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

# # #训练次数
# epochs1 = 100

# sumloss = 0
# for epoch in range(epochs1):
#     sumloss = 0
#     for index in range(len(train_data)):
#         input = train_data[index]
#         output = model(input)
#         targets = test_data[index]
#         loss = criterion(output,targets)
#         sumloss +=loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(epoch)
#     print(sumloss)

# torch.save(model.state_dict(), 'magpie_reverse.pth')
# for i in range(10):
#     input = train_data[i]
#     # print(input)
#     output = model(input)
#     output = output.detach().numpy()
#     # print(output)
#     mmmm = vector_to_chemical_formula(output, elements_order, element_weights)
#     print(mmmm)