import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

#-------------------------------画图函数-----------------------------
def plotData(x,y):
    plt.plot(x, y, 'rx')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of city in 10,000s')
    plt.title('Profit of food trucks in different cities')
    plt.show()

data = genfromtxt('data/ex1data2.txt', delimiter=',')
x = data[:, 0:2]    # 取前两列的数据作为X，所以此时有两个θ
y = data[:, 2]
m = len(y)

plotData(x, y)

#-----------------------------均值归一化处理--------------------------
'''
    a = np.array([[1,2],[3,4]])
    np.mean(a)      将二维数组的每个元素相加除以元素个数，有小数
    np.mean(a, axis=0)      计算每一列的均值
    np.mean(a, axis=1)      计算每一行的均值
    
    np.std()
'''
def featureNormalise(x):
    x_norm = x
    mu = np.mean(x, axis=0)     #求出x每一列的均值
    x_norm -= mu
    sigma = np.std(x, axis=0)   #按行计算标准差
    x_norm /= sigma
    return x_norm, mu, sigma

X, mu, sigma = featureNormalise(x)
print(X[:5])
print(mu)
print(sigma)

#给X新增加一列
X = np.column_stack((np.ones(m),x))
print(X[:5])

#------------------------------代价函数J(θ)--------------------------
def computeCostMulti(x, y, theta):
    J = 0
    w = len(y)

    for i in range(m):
        J += float((np.dot(theta.T, X[i, :]) - y[i])) ** 2

    J = J / (2 * m)
    #或者
    #J = 1 / (2 * m) * np.dot((np.dot(X, theta) - y).T, (np.dot(X, theta) - y))  #两个矩阵的乘法可以代替一个for循环
    return J

theta = np.zeros((3,1))
y = y.reshape((m,1))     #把数据变成m行1列的
#print(computeCostMulti(X, y, theta))

#-----------------------------------梯度下降函数--------------------------------
'''
    np.shape[0]  : 表示矩阵的行数
    np.shape[1]  : 表示矩阵的列数
'''
def gradientDescentMulti(x, y, theta, alpha, num_iters):
    m = len(y)
    num_features = np.shape(theta)[0]      # θ矩阵是个列向量，所以行数表示了特征的个数
    J_history = np.zeros((num_iters, 1))   #

    for interation in range (num_iters):
        delta = np.zeros((num_features, 1))
        for feature in range(num_features):
            for entry in range(m):
                delta[feature] += (np.dot(theta.T, X[entry,:]) - y[entry]) * X[entry, feature]
            delta[feature] = delta[feature] / m

        theta = theta - alpha * delta
        J_history[interation] = computeCostMulti(X, y, theta)

    return theta,J_history

alpha = 0.03
iterations = 400
theta = np.zeros((3,1))
theta , J_history = gradientDescentMulti(X, y, theta, alpha, iterations)

#-------------------------------------打印h(x)-----------------------------------

plt.plot(J_history,'b-')
plt.xlabel('Iteration')
plt.ylabel('Cost J')
plt.title('Convergence of cost function J',y = 1.05)   # y=1.05 表示标题距离表格高1.05
plt.show()

#---------------------------------预测价格---------------------------
'''
    np.insert(arr,obj,values,axis)
    arr原始数组，obj插入元素位置，values是插入内容，axis是按行按列插入
'''
sample_x = np.array([1650.,3.])   #预测 x1=1650,x2=3 时，y的值
sample_x -= mu
sample_x /= sigma       #做的是均值归一化
sample_x = np.insert(sample_x,0,1.)  #补 1

price = float(np.dot(theta.T,sample_x))
print('Predicted price of a 1650 sq-ft, 3 br house is $%.0f.', price)

#------------------------------不同步长下绘制梯度下降函数-----------------------------
theta = np.zeros((3, 1))
iterations = 400
theta = np.zeros((3, 1))
alphas = [0.3, 0.1, 0.03, 0.01, 0.003]

J_histories = []
for alpha in alphas:
    i = alphas.index(alpha)
    J_histories.append(gradientDescentMulti(X, y, theta, alpha, iterations)[1])

for i in range (len(J_histories)):
    plt.plot(J_histories[i])
plt.legend(alphas)
plt.title('Convergence of cost function for different learning rates', y = 1.05)
plt.xlabel('Iteration')
plt.ylabel('Cost J')
plt.show()


