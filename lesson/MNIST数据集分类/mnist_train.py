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
EPOCHS = 10
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

def run1():
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
            loss = functional.cross_entropy(out, y_one_hot)
            
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


'''
构造全连接神经网络进行优化
'''

def run2():
   # w1.shape = out * in(784)
    w1 = torch.randn(256, 784, requires_grad = True)
    b1 = torch.zeros(1, 256, requires_grad = True)
    w2 = torch.randn(256, 256, requires_grad = True)
    b2 = torch.zeros(1, 256, requires_grad = True)
    w3 = torch.randn(10, 256, requires_grad = True)
    b3 = torch.zeros(1, 10, requires_grad = True)

    '''
    何凯明初始化方法
    '''
    torch.nn.init.kaiming_normal_(w1)
    torch.nn.init.kaiming_normal_(w2)
    torch.nn.init.kaiming_normal_(w3)
    
    def forward(x): # x.shape = 1 * 784
        x = functional.relu(x @ w1.t() + b1) 
        x = functional.relu(x @ w2.t() + b2)
        x = x @ w3.t() + b3
        return x # x.shape = 1 * 10 
    
    optimizer =  torch.optim.Adam([w1, b1, w2, b2, w3, b3], lr = LEARNING_RATE)
    criteon = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28*28)
            logits = forward(data)
            loss = criteon(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
    
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)
            logits = forward(data)
            test_loss += criteon(logits, target).item()
    
            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()
    
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 0

def run3():
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()      
            self.model = nn.Sequential(
                nn.Linear(784, 200),
                nn.ReLU(inplace = True),
                nn.Linear(200, 200),
                nn.ReLU(inplace = True),
                nn.Linear(200, 10)
            )
        def forward(self, x):
            x = self.model(x)
            return x
    
    net = MLP()
    optimizer = optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)
    criteon = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28*28)
            logits = net(data)
            loss = criteon(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
    
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)
            logits = net(data)
            test_loss += criteon(logits, target).item()
    
            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()
    
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
run3()