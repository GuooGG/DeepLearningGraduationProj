'''
构造残差模块和残差神经网络
'''
import torch
from torch import nn
from torch.nn import functional as F

'''
残差模块设计
'''
class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride = 1):
        super(ResBlock, self).__init__()
        #卷积层C1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = stride, padding = 1)
        #标准化B2
        self.bn1 = nn.BatchNorm2d(ch_out)
        #卷积层C3
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1)
        #标准化B4
        self.bn2 = nn.BatchNorm2d(ch_out)
        
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(ch_out)
            )
    
    def forward(self, x):
        #x[batch_size, channel, h, w]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.extra(x)
        return out
    
'''
构造18层残差神经网络
'''
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        
        #第一层卷积标准化
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
        )
        self.blk1 = ResBlock(64, 128, stride = 2)
        self.blk2 = ResBlock(128, 256, stride = 2)
        self.blk3 = ResBlock(256, 512, stride = 2)
        self.blk4 = ResBlock(512, 512)
        
        self.outlayer = nn.Linear(8192,10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x
