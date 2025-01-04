'''
构建感知机 y = ΣWx + b
'''
import torch
from torch.nn import functional as F

'''
单层单输出
'''
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad = True)

o = torch.sigmoid(torch.matmul(x, w.t()))
print(o)

loss = F.mse_loss(o, torch.ones(1,1))
loss.backward()
print(w.grad)

'''
单层多输出 10 输入,3输出  
'''
x = torch.randn(1, 10)
w = torch.randn(3, 10, requires_grad = True)
o = torch.sigmoid(torch.matmul(x, w.t()))
loss = F.mse_loss(o, torch.ones(1, 3))
loss.backward()
print(w.grad)


'''
多层单输出
'''
x = torch.randn(1, 10)
w1 = torch.randn(3, 10, requires_grad = True)
b1 = torch.randn(1, 3, requires_grad = True)
w2 = torch.randn(1, 3, requires_grad = True)
b2 = torch.randn(1, 1, requires_grad = True)

y1 = F.sigmoid(x @ w1.t() + b1) #1 3
y2 = F.sigmoid(y1 @ w2.t() + b2) # 1 1

loss = F.mse_loss(y2, torch.ones(1, 1))
loss.backward()
print(w1.grad)
print(b1.grad)
print(w2.grad)
print(b2.grad)  

