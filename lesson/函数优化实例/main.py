'''
求以下函数最小值所在位置
f(x, y) = (x^2 + y -11)^2 + (x + y^2 - 7)^2
'''

import torch

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def function(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


'''
画出函数图像
'''
def draw():
    '''
    生成网格
    '''
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    z = function([X, Y])

    '''
    构建画布
    '''
    fig = plt.figure("2d function figure")
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X, Y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
'''
解极值
'''
x = torch.tensor([15., -15.], requires_grad = True) # x = 3 y = -2

'''
构造优化器
'''
optimizer = torch.optim.SGD([x], lr=0.001, momentum=0.9)


for step in range(10001):
    
    # 预测值
    pred = function(x)
    #计算梯度
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()
    
    if step % 2000 == 0:
        print("当前迭代{0}次,x = {1}, y = {2}, f(x,y) = {3}".format(step,x[0], x[1], pred))
    
    
    
    
