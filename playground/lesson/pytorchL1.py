import torch
import time
""""
矩阵相乘
"""
print(torch.__version__)
print(torch.cuda.is_available())
a = torch.randn(10000,1000)
b = torch.randn(1000,2000)

t0 = time.time()
c = torch.matmul(a,b)
t1 = time.time();

print(t1-t0)

""""
求导
"""
from torch import autograd
x = torch.tensor(1.) 
a = torch.tensor(1., requires_grad= True)
b = torch.tensor(2., requires_grad= True)
c = torch.tensor(3., requires_grad= True)

y = a**2 * x + b * x + c
print("before", a.grad,b.grad,c.grad)
grads = autograd.grad(y,[a,b,c])
print("after",grads[0],grads[1],grads[2])

