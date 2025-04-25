#用来训练从隐变量到T,P,C的映射
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from GET_latent import *
from GET_TPC import *
import os
import torch.nn as nn
from transform import *
from get_target import *
from sklearn.preprocessing import StandardScaler
import joblib
# # # 初始化标准化器
# file_path2 = "Composition control absorption1_magpie2.xlsx"
# file_path1 = "latent_vectors.xlsx"
# scaler = StandardScaler()
# # # # 训练数据
# test_data = GET_IN(file_path2)
# train_data = GET_FOR(file_path1)
# test_data = scaler.fit_transform(test_data)
# test_data = torch.tensor(test_data).float()
# # joblib.dump(scaler, 'scaler.joblib')
# scaler = joblib.load('scaler.joblib')

class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
        )
    def forward(self, x):
        return self.model(x)



# # # #开模型
# model = My_model()
# # model.load_state_dict(torch.load('latent_to_TPC.pth'))
# # y = torch.tensor([-3.0952, -0.4277, 2.9728])

# # y = y.detach().numpy()  # 分离梯度并转为 NumPy
# # y = scaler.transform(y.reshape(1,-1))
# # y = torch.tensor(y).float()
# # TPC = model(y)
# # print(TPC)
# # #损失函数
# criterion = nn.MSELoss()

# # # 定义L1正则化系数
# # lambda_l1 = 0.00001
# # #优化器
# learing_rate = 0.0001
# # #梯度裁剪
# # clip_value = 0.0001
# # #读取保存的权重
# # # file_name ='laten_to_model.pth'
# # # state_dict = torch.load(file_name)
# # # model.load_state_dict(state_dict)

# # #梯度下降函数
# # # optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

# # #训练次数
# epochs1 = 20


# sumloss = 0
# for epoch in range(epochs1):
#     sumloss = 0
#     for index in range(len(train_data)):
#         input = train_data[index]
#         output = model(input)
#         targets = test_data[index]
#         # targets = targets.unsqueeze(0)
#         # print(index,targets)
#         # print(index,output)
#         loss = criterion(output,targets)
#         sumloss += loss 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(epoch)
#     print(sumloss)


# # for i in range(10):
# #     print(i)
# #     x = train_data[i]
# #     x = x.reshape(1,-1)
# #     x = scaler.transform(x)
# #     x = torch.tensor(x).float()
# #     y = model(x)
# #     # y = y.reshape(1,-1)
# #     # y = y.detach().numpy()
# #     # y = scaler.inverse_normalize(y, scaler)
# #     print(y)
    
# #     print(test_data[i])

# torch.save(model.state_dict(), 'latent_to_TPC.pth')







