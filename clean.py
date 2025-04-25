import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def get_dataset(file_path):
    file_path = "Composition control absorption1_magpie2.xlsx"
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df.iloc[:, 3:7]
    # 将 Pandas 数据框转换为 NumPy 数组
    data_numpy = df.values
    # 将 NumPy 数组转换为 PyTorch 张量
    data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
    print("PyTorch 张量:")
    print(data_tensor)
    print("张量形状:", data_tensor.shape)
    return data_tensor

if __name__ == "__main__":
    file_path = "Composition control absorption1_magpie2.xlsx"
    data_tensor = get_dataset(file_path)
    print(data_tensor)