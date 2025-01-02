import torch
from torch import nn
from torch.nn import functional
from torch import optim
import torch.utils
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt

BATCH_SZIE = 512
LEARNING_RATE = 0.01
'''
实现对标签的ont-hot编码
'''
def one_hot(label, depth = 10):
    out = torch.zeros((label.size(0), depth))
    out.scatter_(1, label.unsqueeze(1), 1)
    return out


'''
绘制折线图
'''
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

'''
画图
'''
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


optimizer = optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = 0.9)

'''
记录目标函数的值
'''
train_loss = []

'''
开始训练,训练三轮
'''
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 28*28)
        out = net(x)
        y_one_hot = one_hot(y)
        loss = functional.mse_loss(out, y_one_hot)
        
        if batch_idx % 10 == 0:
            print("now loss = {0}".format(loss.item()))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())      
   
'''
画出梯度下降过程曲线
''' 
plot_curve(train_loss)

'''
测试模型正确率
'''
total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim = 1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("模型正确率为{0}%".format(acc*100))


x,y = next(iter(test_loader))
out = net(x.view(x.size(0),28*28))
pred = out.argmax(dim = 1)
plot_image(x, pred, "test")