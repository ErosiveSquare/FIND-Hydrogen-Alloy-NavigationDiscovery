#用来获取隐变量
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import openpyxl
def GET_FOR(file_path):
        file_path = "latent_vectors.xlsx"
        df = pd.read_excel(file_path, engine='openpyxl')
        df = df.iloc[:,0:3]
        # 将 Pandas 数据框转换为 NumPy 数组
        data_numpy = df.values
        data_numpy = data_numpy.reshape(2938, -1)
        # 将 NumPy 数组转换为 PyTorch 张量
        data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
        return data_tensor

if __name__ == "__main__":
        file_path = "output.xlsx"
        data_tensor = GET_FOR(file_path)
        # x = data_tensor[0]
        # x = x.reshape(x.shape[0], -1)
        print(data_tensor[0])
