#获取化学式
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

def GET_X(file_path):
    file_path = "Composition control absorption1_magpie2.xlsx"
    df = pd.read_excel(file_path, usecols=[1])
    # 将 Pandas 数据框转换为 NumPy 数组
    return df


if __name__ == "__main__":
    file_path = "Composition control absorption1_magpie2.xlsx"
    data_tensor = GET_X(file_path)
    print(data_tensor)

