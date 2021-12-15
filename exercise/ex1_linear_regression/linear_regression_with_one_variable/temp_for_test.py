import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

theta = np.matrix(np.array([0, 0]))  # theta是一个1行2列的矩阵且都是0

parameters = (theta.shape[1])  # ravel()函数计算需要求解theta的参数个数，即2
print(parameters)
