import torch
from torch import nn
from torch.nn import functional as Fun

class Lenet5(nn.Module):
    #构造函数
    def __init__(self):
        super(Lenet5, self).__init__()
        '''
        构造网络
        '''
        self.conv_unit = nn.Sequential(
            # 输入为3通道，输出6个通道,卷积核大小为5*5,一次移动一格,最外围padding宽度为0
            # 输入[batch_size, 3, 32, 32] 输出[batch_size, 6, 28, 28] 32-5+1=28
            nn.Conv2d(3, 6, kernel_size = 5, stride = 1, padding = 0),
            # 池化层,平均池化,不改变通道数量，只改变尺寸大小
            # 输入[batch_size, 6, 28, 28] 输出[batch_size, 6, 14, 14] 28/2=14
            nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),
            #卷积层2
            #输入[batch_size, 6, 14, 14] 输出[batch_size, 16, 10, 10]
            nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 0),
            #平均池化层2
            #输入[batch_size, 16, 10, 10] 输出[batch_size, 16, 5, 5]
            nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),
            #全连接层,连接之前应预先打平,但Sequential不提供打平的功能
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batch_size, -1) # 使用-1将所有图片全部打平 [32 , 16*5*5]
        logits = self.fc_unit(x) 
        return logits              #[batch_size, 10]
            
            
