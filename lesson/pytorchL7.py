'''
Entropy = -ΣP(i)*logP(i) 
'''
import torch
from torch.nn import functional as F


p1 = torch.full([4], 0.25)
p2 = torch.tensor([0.8, 0.1, 0.05, 0.05])
print(-(p1 * torch.log(p1)).sum())
print(-(p2 * torch.log(p2)).sum())

'''
交叉熵损失函数 Cross Entropy Loss
H(P,Q) = -ΣP(x)*logQ(x) 
'''
x = torch.randn(1,784)
w = torch.randn(10, 784) 
o = torch.matmul(x, w.t()) # 1 * 10
pred = F.softmax(o, dim = 1)
print(pred.sum())

#corss_entropy已经包含了softmax步骤，不可以放入softmax处理后的数值
h =  F.cross_entropy(o, torch.tensor([3]))
print(h)
print(F.cross_entropy(torch.tensor([[1.,1.]]), torch.tensor([0])))