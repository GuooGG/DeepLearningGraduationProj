''''
创建tensor(张量)数据类型
'''
import torch

'''
创建一个2*3的张量,内部元素为服从N(0,1)随机数
'''
m_tensor = torch.randn(2, 3)
print(m_tensor)
print(m_tensor.type())

'''
数据部署位置
'''
# m_tensor = m_tensor.cuda()
print(isinstance(m_tensor, torch.cuda.FloatTensor))

'''
标量
'''
x = torch.tensor(1.)
y = torch.tensor(2)
z = torch.tensor(3.6)
print(x,y,z)
print(x.type(),y.type(),z.type())
'''维度为0的张量可以是标量'''
print(x.dim(),x.shape)

'''维度为2的张量可以是向量'''
x = torch.tensor([1,1])
y = torch.tensor([1,2.6,2.4])
z = torch.tensor([1.1])
print(x,y,z)
print(x.dim(),x.shape)
x = torch.FloatTensor(1,2,3)
print(x,x.dim())
x = torch.randn(4,3,2,1)
print(x, x.dim())
print(x.shape)

'''
一个四维张量,512张图片,3通道彩色图片,像素为28*28  
'''
patch = torch.randn(512,3,28,28)

'''
计算tensor的大小
'''
print(patch.numel())


