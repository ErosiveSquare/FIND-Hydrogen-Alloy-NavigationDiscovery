#用来训练从T,P,C到隐变量的映射
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from GET_latent import *
from GET_TPC import *
import os
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
# scaler = StandardScaler()

# # # 初始化标准化器
# file_path2 = "Composition control absorption1_magpie2.xlsx"
# file_path1 = "latent_vectors.xlsx"

# # # # # 训练数据
# train_data = GET_IN(file_path2)
# print(train_data[8])
# test_data = GET_FOR(file_path1)
# train_data = scaler.fit_transform(train_data)
# train_data = torch.tensor(train_data).float()
# # joblib.dump(scaler, 'scaler2.joblib')
class My_model2(nn.Module):
    def __init__(self):
        super(My_model2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,6),
            nn.ReLU(),
            nn.Linear(6,3),
        )
    def forward(self, x):
        return self.model(x)



# #开模型
# model2 = My_model2()

# # # #损失函数
# criterion = nn.MSELoss()

# # # # 定义L1正则化系数
# # # lambda_l1 = 0.00001
# # # #优化器
# learing_rate = 0.001
# # # #梯度裁剪
# # # clip_value = 0.0001
# # # #读取保存的权重
# file_name ='TPC_to_latent.pth'
# state_dict = torch.load(file_name)
# model2.load_state_dict(state_dict)

# # #梯度下降函数
# # # optimizer = torch.optim.SGD(model2.parameters(), lr=learing_rate)
# optimizer = torch.optim.Adam(model2.parameters(), lr=learing_rate)

# # # #训练次数
# # epochs1 = 20

# # sumloss = 0
# # for epoch in range(epochs1):
# #     sumloss = 0
# #     for index in range(len(train_data)):
# #         input = train_data[index]
# #         output = model2(input)
# #         targets = test_data[index]
# #         # targets = targets.unsqueeze(0)
# #         # print(index,targets)
# #         # print(index,output)
# #         loss = criterion(output,targets)
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
# #         sumloss += loss
# #     print(sumloss)

# for i in range(10):
#     print(i)
#     x = train_data[i]
#     y = model2(x)
#     # y = y.reshape(1,-1)
#     # y = y.detach().numpy()
#     # y = scaler.inverse_normalize(y, scaler)
#     print(y)
#     print(test_data[i])

# # torch.save(model2.state_dict(),'TPC_to_latent.pth')


