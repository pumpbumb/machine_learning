import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 将数据初步处理
path = 'ex1_data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.insert(0, 'Ones', 1)  # 在第1列前面再插入1列且全为1，列名为’Ones’，用于更新θo
print(data)
