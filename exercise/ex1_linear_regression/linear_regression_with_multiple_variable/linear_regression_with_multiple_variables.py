import numpy as np
import pandas as pd


# 计算J(θ)，X是矩阵
def compute_cost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 实现了θ的更新
def gradient_descent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        cost[i] = compute_cost(X, y, theta)
    return theta, cost


# 读取数据
path = 'ex1_data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std()  # 特征归一化
data2.insert(0, 'Ones', 1)  # 添加每个训练实例的第一个特征为1

# 初始化X和y
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols - 1]
y2 = data2.iloc[:, cols - 1:cols]

# 转换成matrix格式，初始化theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

# 初始化一些附加变量: 学习速率α和要执行的迭代次数
alpha = 0.01
iters = 1500

# 运行梯度下降算法
g2, cost2 = gradient_descent(X2, y2, theta2, alpha, iters)
print('打印最终的theta矩阵:\n', g2)
print('每一次迭代损失函数的值:\n', cost2)
