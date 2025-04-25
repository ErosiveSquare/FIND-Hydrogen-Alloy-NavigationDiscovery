#把化学式转变为向量
import re
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import openpyxl


# 定义元素顺序列表
# 定义元素顺序列表（40个元素）
elements_order = [
    'Ti', 'Ni', 'Mn', 'Mg', 'Cr', 'Fe', 'La', 'Zr', 'V', 'Y',
    'Al', 'Co', 'Ce', 'Sm', 'Cu', 'Nd', 'Ca', 'Pr', 'Nb', 'Mo',
    'Gd', 'In', 'Sn', 'Ag', 'Sc', 'Zn', 'Er', 'Hf', 'Pd', 'Re',
    'Bi', 'Ga', 'W', 'Rh', 'Ho', 'Pb', 'Na', 'Ta', 'Cd', 'Ru'
]

# 定义元素权重字典（40个元素，实际使用时应替换为真实的原子量）
element_weights = {
    'Ti': 47.867, 'Ni': 58.693, 'Mn': 54.938, 'Mg': 24.305, 'Cr': 51.996, 'Fe': 55.845, 'La': 138.91, 'Zr': 91.224, 'V': 50.942,
    'Y': 88.906, 'Al': 26.982, 'Co': 58.933, 'Ce': 140.12, 'Sm': 150.36, 'Cu': 63.546, 'Nd': 144.24, 'Ca': 40.078, 'Pr': 140.91, 'Nb': 92.906,
    'Mo': 95.95, 'Gd': 157.25, 'In': 114.82, 'Sn': 118.71, 'Ag': 107.87, 'Sc': 44.956, 'Zn': 65.38, 'Er': 167.26, 'Hf': 178.49, 'Pd': 106.42, 'Re': 186.21,
    'Bi': 208.98, 'Ga': 69.723, 'W': 183.84, 'Rh': 102.91, 'Ho': 164.93, 'Pb': 207.2, 'Na': 22.990, 'Ta': 180.95, 'Cd': 112.41, 'Ru': 101.07
}

def chemical_formula_to_vector(formula, elements_order, element_weights):
    # 定义正则表达式模式以匹配元素和数量
    pattern = r'([A-Z][a-z]*)(\d*\.\d+|\d+)'
    matches = re.findall(pattern, formula)

    # 初始化一个字典来存储元素及其总重量
    element_weights_vector = {element: 0.0 for element in elements_order}

    for element, count in matches:
        if count == '':
            count = 1
        else:
            count = float(count)
        if element in element_weights_vector and element in element_weights:
            element_weights_vector[element] += element_weights[element] * count

    # 将字典转换为向量
    vector = [element_weights_vector[element] for element in elements_order]

    return vector




# 示例化学式
# formula = "Ti0.16Cr0.3V0.5Nb0.04"

# # 将化学式转换为向量
# vector = chemical_formula_to_vector(formula, elements_order, element_weights)
# print(vector)


def GET_X(file_path):
    # 读取Excel文件
    file_path = "Composition control absorption1_magpie2.xlsx"
    df = pd.read_excel(file_path)
    # 假设化学公式列的名称是 'Formula'
    formulas = df['Components2'].astype(str)
    vectors = formulas.apply(lambda x: chemical_formula_to_vector(x, elements_order, element_weights))
    vectors = torch.tensor(vectors)
    return vectors

# if __name__ == "__main__":
#     file_path = "Composition control absorption1_magpie2.xlsx"
#     data_tensor = GET_X(file_path)
#     print(data_tensor)