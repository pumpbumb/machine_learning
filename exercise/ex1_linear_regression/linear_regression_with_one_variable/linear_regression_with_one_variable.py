'''
# 在这里，训练集、theta都是行向量的形式，假设函数是用x乘上theta的转置。
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 计算J(θ)，X是训练集矩阵，会从data中取出: X = data.iloc[:, :-1]，亦即X是data里的除最后列
def compute_cost(X, y, theta):
    # power(x, y) 函数，计算 x 的 y 次方。
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 实现了θ的更新
def gradient_descent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))  # 创建1行2列的0矩阵
    parameters = int(theta.ravel().shape[1])  # ravel()函数计算需要求解theta的参数个数，即2.
    cost = np.zeros(iters)  # 创建iters个0的数组
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))  # theta1和2更新
        theta = temp
        cost[i] = compute_cost(X, y, theta)
    return theta, cost


# 将数据初步处理
path = 'ex1_data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# 让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。
data.insert(0, 'Ones', 1)  # 在原第一列再插入1列且全为1，列名为’Ones’，用于更新θo

# 画出训练集的数据图
# figsize = (a, b)，其中figsize(即 figure size 的缩写)用来设置图形的大小，
# a为图形的宽， b为图形的高，单位为英寸。
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()

# 初始化X和y
cols = data.shape[1]  # 获取data的列数
X = data.iloc[:, :-1]  # X是data里的除最后列
y = data.iloc[:, cols - 1:cols]  # y是data最后一列

# 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。
# 我们还需要初始化theta。
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))  # theta是一个1行2列的矩阵且都是0
# print(X.shape, theta.shape, y.shape)  # 查看X y θ 的维度
# print(computeCost(X, y, theta))  # 计算代价函数(θ初始值为0)，答案应该是32.07

# 初始化一些附加变量: 学习速率α和要执行的迭代次数
alpha = 0.01
iterations = 1500

# 现在让我们运行梯度下降算法来将我们的参数θ适合于训练集。
g, cost = gradient_descent(X, y, theta, alpha, iterations)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)  # 最终的假设函数

# 原始数据以及拟合的直线
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

'''
# 预测35000和70000城市规模的小吃摊利润
predict1 = [1, 3.5] * g.T
print("predict1:", predict1)
predict2 = [1, 7] * g.T
print("predict2:", predict2)
'''
