import torch
from torch import nn
from torch.nn import functional
from torch import optim
import torch.utils
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt

BATCH_SZIE = 512
LEARNING_RATE = 0.001
'''
实现对标签的ont-hot编码
'''
def one_hot(label, depth = 10):
    out = torch.zeros(label.size(), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim = 1, index = idx, value = 1)
    return out

'''
加载训练集
'''
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        'mnist_data', train = True, download = True,
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.13,), (0.31)
            )
        ])
    ),
    batch_size = BATCH_SZIE, shuffle = True
)

'''
加载测试集
'''
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        'mnist_data/', train = False, download = True, 
        transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.13,), (0.31,)
            )
        ])
    ),
    batch_size = BATCH_SZIE, shuffle = True
)

'''
定义神经网络类
'''
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        '''
        构建三层网络,第一层输入为28*28,第三城输出为10*1
        '''
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    
    '''
    神经网络前向传播过程
    '''
    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
实例化神经网络对象
'''
net = Net()

def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

x, y = next(iter(iter(train_loader)))
plot_image(x,y,"image_sample")

