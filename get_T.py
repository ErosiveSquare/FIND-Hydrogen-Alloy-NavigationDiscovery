#用来获取T,P,C
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import openpyxl
def get_T(file_path):
        file_path = "Composition control absorption1_magpie2.xlsx"
        df = pd.read_excel(file_path, usecols=[3])
        # 将 Pandas 数据框转换为 NumPy 数组
        data_numpy = df.values
        data_numpy = data_numpy.reshape(2938,-1)
        # 将 NumPy 数组转换为 PyTorch 张量
        data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
        return data_tensor

if __name__ == "__main__":
        file_path = "Composition control absorption1_magpie2.xlsx"
        data_tensor = GET_IN(file_path)
        print(data_tensor)