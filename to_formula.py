import re
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


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

def vector_to_chemical_formula(vector, elements_order, element_weights):
    # 将向量转换为元素及其总重量的字典
    element_weights_dict = {element: weight for element, weight in zip(elements_order, vector)}

    # 计算每个元素的数量
    formula_parts = []
    for element in elements_order:
        if element in element_weights and element_weights[element] != 0:
            total_weight = element_weights_dict[element]
            if total_weight > 0:
                count = total_weight / element_weights[element]
                if count > 0:
                    # 保留小数点后两位
                    count_str = f"{count:.2f}"
                    formula_parts.append(f"{element}{count_str}")

    # 生成化学式字符串
    formula = ''.join(formula_parts)
    return formula


# 示例向量
# vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#           16 * 47.867, 0.0, 0.0, 30 * 51.996, 0.0, 0.0, 0.0, 50 * 50.942, 0.0, 4 * 92.906,
#           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#           0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#
# # 将向量转换为化学式
# formula = vector_to_chemical_formula(vector, elements_order, element_weights)
# print(formula)