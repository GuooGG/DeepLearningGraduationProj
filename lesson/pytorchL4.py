'''
操作张量类型
'''
import torch

a = torch.randn(5, 3, 28, 28)

print(a[0])
print(a[0].shape) # 3 28 28
print(a[1])
print(a[1].shape) # 3 28 28

print(a[2][1]) # 第三张图的第二个通道
print(a[2][1].shape) # 28 28

print(a[0][0][27][27]) # 取一个像素点上红色通道的值 （ dim = 0）

print(a[0:2].shape) # 2 3 28 28 取前两张图片
print(a[0:5,0:2].shape) # 5 2 28 28 取五张图片的前两个通道信息


x = torch.randn(3, 4)
mask = x.ge(0.5) # 创建一个3*4矩阵，大于等于0.5的元素置为1
print(torch.masked_select(x, mask)) # 取出大于0.5的元素，打印出一个一维向量

s = torch.tensor([[4, 3, 5],
                 [2, 6, 7]])
print(torch.take(s, torch.tensor([0, 3, -2]))) # 打平后取出0 ,3, -2个元素 [4, 2, 6]

'''
对张量进行变形
'''
a = torch.randn(4, 1, 28, 28)
print(a.shape)
print(a.view(4, 1, 28 * 28).shape)
print(a.view(4 * 1, 28 * 28).shape)
print(a.view(4 * 1 * 28 * 28).shape)

'''
升维，降维
'''
a = torch.randn(4, 1 , 1, 28, 28)
print(a.shape)
print(a.squeeze().shape) # squeeze降维， 将size为1的维度删除

print(a.squeeze(1).shape) # 若第二个维度size为1，则删除第二个维度
print(a.unsqueeze(0).shape) # 在第一个维度之前增加一个size为1的维度

'''
扩张
'''
x = torch.rand(32).unsqueeze(0).unsqueeze(2).unsqueeze(2)
print(x.shape)
print(x.expand(4, 32, 14, 14).shape)


'''
四张图片32个通道,为每张图片加上x的像素偏移
'''
x = torch.rand(32)
f = torch.rand(4, 32, 14, 14)
x = x.unsqueeze(0).unsqueeze(2).unsqueeze(2).expand(4, 32, 14, 14) # 4 32 14 14
f +=  x

'''
转置
'''
a = torch.rand(3, 4)
print(a.t().shape) # 对二维张量，用.t()转置
a = torch.rand(1, 2, 3, 4, 5)
print(a.transpose(0, 3).shape) # 第一个维度于第四个维度交换
print(a.permute(4, 3, 2, 1, 0).shape) # 对多维张量多个维度进行转置
'''
深拷贝
'''
b = a.contiguous()
print(b.shape)











