import torch
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from latent_to_TPC import *
from TPC_to_latent import *
from magpie_re import *
from to_formula import *
from transform import *
from test5 import *
from pymatgen.core import Composition
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition


def process_chemical_formula(formula_str):
    # 使用正则表达式匹配元素和对应的数值
    elements = re.findall(r'([A-Z][a-z]*)(\d+\.\d{2})', formula_str)
    processed = []
    for elem, val_str in elements:
        # 将数值转换为浮点数，乘以100后四舍五入取整
        value = round(float(val_str) * 100)
        if value < 1.1:
            continue  # 过滤掉数值为0的元素
        processed.append(f"{elem}{value}")
    # 组合成最终的化学式字符串
    return ''.join(processed)

vae = VAE(133,3)
vae.load_state_dict(torch.load('vae_.pth'))

laten_to_ = My_model()
laten_to_.load_state_dict(torch.load("latent_to_TPC.pth"))

TPC_to_latent = My_model2()
TPC_to_latent.load_state_dict(torch.load("TPC_to_latent.pth"))

magpie_reverse = My_model3()
magpie_reverse.load_state_dict(torch.load('magpie_reverse.pth'))


def TPC_to_formula(T,P,C):

    scaler2 = joblib.load('scaler2.joblib')
    print("TPC:")
    print(T, P, C)

    TPC = torch.tensor([T, P, C], dtype=torch.float32)

    #scaler3 = joblib.load('scaler3.joblib')
    TPC = torch.tensor(TPC)
    TPC = TPC.detach().numpy()
    TPC = scaler2.transform(TPC.reshape(1,-1))
    TPC = torch.tensor(TPC).float()
    y = TPC_to_latent(TPC)
    #print(y)
    y = vae.decode(y)
    #print(y)


    y = y.detach().cpu().numpy()  # 分离梯度并转为 NumPy
    #y = scaler3.transform(y)
    y = torch.tensor(y).float()
    y = y[0]
    y = y[:132]
    y = torch.tensor(y)
    #print(y.shape)
    y = magpie_reverse(y)
    y = y.detach().cpu().numpy()  # 分离梯度并转为 NumPy
    formula_pred = vector_to_chemical_formula(y, elements_order, element_weights)
    print(formula_pred)
    rt = process_chemical_formula(formula_pred)
    print("清洗后：")
    print(rt)
    return rt



