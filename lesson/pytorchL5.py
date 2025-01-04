'''
张量之间的运算 broadcasting
'''
import torch
from torch.nn import functional as F

a = torch.randn(8)
print(a.unsqueeze(0).unsqueeze(0).expand(4, 32, 8).shape)

'''
拼接和拆分 cat 
'''
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
'''
cat:中括号填入要合并的张量,并在指定dim合并,其他dim必须shape相同
'''
print(torch.cat([a, b], dim = 0).shape) 

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
a3 = torch.rand(4, 1, 32, 32)
print(torch.cat([a1, a3], dim = 1).shape) # RGB->RGBA

'''
stack拼接,增加新的维度.两个张量应该有完全相同的shape
'''
a = torch.rand(32, 8)
b = torch.rand(32, 8)
print(torch.stack([a, b], dim = 0).shape)


'''
拆分 split按长度拆分
'''
c = torch.rand(4, 32, 8)
aa, bb = c.split([3, 1],dim = 0)
print(aa.shape, bb.shape)


'''
chunk按数量拆分
'''
c = torch.rand(8, 32, 8)
aa, bb, cc= c.chunk(3, dim = 0)
print(aa.shape, bb.shape, cc.shape)

'''
基本运算 +-*/^
'''

a = torch.rand(3, 5)
b = torch.rand(3, 5)
print(a + b)
print(torch.all(torch.eq(a + b, torch.add(a, b))))

''''
均为 3 * 5 
'''
print(torch.add(a, b))
print(torch.sub(a, b))
print(torch.mul(a, b))
print(torch.div(a, b))

a = torch.rand(3, 5)
b = torch.rand(5)
print(a, b)
print(a + b)

'''
张量乘法 除了最后两个维度,其他所有维度shape相同,最后两个维度满足矩阵相乘条件
'''
a = torch.full([1, 2], 3.)
b = torch.ones(2, 8)
print(torch.matmul(a, b))

a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
print(torch.matmul(a, b).shape)

c = torch.rand(4, 1, 64, 32)
print(torch.matmul(a, c).shape)

'''
梯度裁剪(梯度离散，梯度爆炸)
'''
grad = torch.rand(2, 3) * 10
print(grad.max())
print(grad.clamp(1, 5))

'''
获取张量的特征,求p范数
'''
a = torch.full([8], 1.)
print(a.norm(1), a.norm(2), a.norm(3))
a = torch.full([1, 2, 3], 1.)
print(a.norm(1), a.norm(2), a.norm(3))
'''
指定维度上的范数
'''
print(a.norm(1, dim = 1), a.norm(1, dim = 0).shape)

'''
统计属性
'''
x = torch.arange(8.).view(2, 4)
print(x.min(),x.max())
print(x.mean())
print(x.prod())
print(x.sum())
print(x.argmax(dim = 0)) # 1 1 1 1

x = torch.rand(4, 10)
print(x.argmax(dim = 1))
print(x.argmax(dim = 1, keepdim = True))
'''
topk 
'''
print(x.topk(3, dim = 1))

'''
比较
'''
print(x > 2)

a = torch.rand(2, 3)
b = torch.ones(2, 3)
print(torch.eq(a, b))
print(torch.equal(a, b))

'''
gather
'''
prob = torch.randn(4, 10)
idx = prob.topk(3, dim = 1)
print(idx)
idx = idx[1]
print(idx)
label = torch.tensor([12, 15, 16 ,17, 18, 6, 7 ,8, 9, 10])
result = torch.gather(label.expand(4, 10), dim = 1, index = idx.long())
print(result)


a = torch.rand(3, requires_grad = True)
p = F.softmax(a, dim = 0)
for i in range(len(p)):
    print(torch.autograd.grad(p[i], [a], retain_graph = True))

