from numpy import *
import matplotlib.pyplot as plt

# 数据形式
'''
x1          x2          y
-0.017612	14.053064	0
-1.395634	4.662541	1
-0.752157	6.538620	0
-1.322371	7.152853	0
'''
data = []
label = []
fr = open('data/testSet.txt')

for line in fr.readlines():
    line = line.strip().split()
    # 添加上y=w1x1+w2x2+b的b
    data.append([1.0, float(line[0]), float(line[1])])
    label.append(int(line[2]))
print(data)
print(data[0])
print(label)


def plot_function(data, label, weights=None):
    n = shape(data)[0]  # shap(100,3)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(n):
        if len(data[i]) < 3 : print(data[i])
        if int(label[i]) == 1:
            x1.append(data[i][1])
            y1.append(data[i][2])
        else:
            x2.append(data[i][1])
            y2.append(data[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=100, c='red', marker='s')
    ax.scatter(x2, y2, s=100, c='black')

    if weights != None:
        x = arange(-3.0, 3.0, 0.1)
        y = -(weights[0] + weights[1] * x) / weights[2]  # x2=(w1*1 + w[0])/w2

    ax.plot(x, y)

    plt.show()




def sigmoid(t):
    return 1.0 / (1 + exp(-t))


data = mat(data)  # matrix 100*3
label = mat(label).transpose()  # 或者.T 转置
m, n = shape(data)  #
alpha = 0.001  # 步长
maxIter = 500  # 最大迭代次数
weights = ones((n, 1))  # 3*1,w1<=>b

for k in range(maxIter):
    y = sigmoid(data * weights)
    l_wx = -label.T*log(y) - (ones((m,1)).T-label.T)*log(1-y)
    print('step',k,'loss',l_wx)
    error = label - y
    grad = data.T * error
    weights += alpha * grad

plot_function(data.tolist(), label, weights.T.tolist()[0])
print(weights.T.tolist()[0])
