import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
#-------------------------------------------画出数据图--------------------------------------------
#画图函数
def plotData(x,y):
    '''
    plt.plot() 函数
        标准形式：plt.plot(x, y, format_string, **kwargs)
        format_string  由颜色字符、风格字符、标记字符组成

        'r' 红色
        'x' x标记
    '''
    plt.plot(x,y,'rx')
    plt.ylabel('Profit in $10,000s')    #纵轴特征
    plt.xlabel('Population of city in 10,000s')    #横轴特征
    plt.title('Profit of food trucks in different cities')    #表头
    plt.show()

#delimiter 表示以，进行分割
data = genfromtxt('data/ex1data1.txt',delimiter=',')
print(data[:5,:])

#data[ a , b ] a的位置限制第几行，b的位置限制第几列
x = data[:, 0]    #第一列所有数据
y = data[:, 1]    #第二列所有数据
m = len(x)        #样本数m  写成m = len(y) 也可
print(x[:5])      #打印出x中0-5的元素，不包括5
print(y)

#plotData(x, y)    #画图

#-----------------------------------代价函数 J(θ)-----------------------------------------
'''
    np.dot(x,y) 是矩阵乘法或者点积
    x: m*n 的矩阵      y: n*m 的矩阵
    矩阵乘法，例如np.dot(X,X.T)。
    点积，比如np.dot([1,2,3],[4,5,6]) = 1*4 + 2*5 + 3*6  = 32
'''
# 代价函数
def computeCost(x, y, theta):
    J = 0
    w = len(y)

    for i in range(m):
        J += float((np.dot(theta.T, x[i, :]) - y[i])) ** 2     # ** 是幂运算， * 是乘法  theta.T 是转置
                                                               # 实际做的是一个点乘运算，x取出来一行
    J = J / (2 * m)
    return J
'''
    np.column_stack()   将两个矩阵按列合并
    np.row_stack()      将两个矩阵按行合并
    np.ones(m)          创建一个长度为m的一维数组，元素默认数据类型float，所以是1.
    np.ones(m,n)        创建一个长度为m*n的二维数组
    np.ones(m,dtype= int)  创建一个数据类型为int 的一维数组 
'''
x = np.column_stack((np.ones(m), x))
print(x[:5, :])
'''
    zeros(shape, dtype=float, order='C')
    返回一个给定形状和类型的用0填充的数组
'''
theta = np.zeros((2, 1))      #返回一个两行一列的数组  h(x) = 0


print(computeCost(x, y, theta))

theta = np.array([[-1],[2]])    #返回一个两行一列的数组，h(x) = -1+2x
print(computeCost(x, y, theta))

#------------------------------------梯度下降算法--------------------------------
theta = np.zeros((2,1))      #为了能让θ转一圈，所以θ开始要赋值为0
itertations = 1500
alpha = 0.01

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters,1))

    for itertations in range (0, num_iters):
        delta = np.zeros((2, 1))
        for j in range(2):  # j 表示的是 θj ,现在只有两个θ，所以是从0-2，不包括 2
            for i in range(m):
                #θ变换多少次是由m决定的，所以i从0-m ,不包括m
                delta[j] += (np.dot(theta.T, x[i,:]) - y[i]) * x[i,j]
            delta[j] = delta[j] / m

        theta = theta - alpha * delta
        J_history[itertations] = computeCost(x, y, theta)

    return theta

theta = gradientDescent(x, y, theta, alpha, itertations)
print(theta)

#-------------------------------打印出h(x)直线------------------------------
plt.plot(x[:,1], y, 'rx')
plt.plot(x[:,1], np.dot(x,theta), 'b-')       # np.dot() 矩阵乘法，这个的出来的是h(x)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of city in 10,000s')
plt.title('Profit of food trucks in different cities')
plt.legend(['Training data','Linear regression'])       #设置画线标记
plt.show()

#----------------------------预测数据 h(x) = XΘ 向量乘法---------------------
predict1 = np.dot(np.array([1,3.5]),theta)   #手工实现补1
print('For population = 35,000, we predict a profit of $%.0f.' % float(predict1*10000))

predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of $%.0f.' % float(predict2*10000))
