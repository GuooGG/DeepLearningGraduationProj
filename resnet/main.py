import torch
from torch import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import transforms
from resnet import ResNet18

'''
常用常量
'''
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10


'''
加载数据集
'''
cifar_trainset = datasets.CIFAR10('cifar_data', train = True,
                                transform = transforms.Compose([
                                    transforms.Resize((32, 32)), # 将图片转为32 *32
                                    transforms.ToTensor(),        # 将数据转为张量
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], # 对输入进行标准正态化
                                                         std = [0.229, 0.224, 0.225])
                                ]), download = True)
cifar_trainset = DataLoader(cifar_trainset, 
                            batch_size = BATCH_SIZE, 
                            shuffle = True)

#加载测试集
cifar_testset = datasets.CIFAR10('cifar_data', train = False,
                                transform = transforms.Compose([
                                    transforms.Resize((32, 32)), # 将图片转为32 *32
                                    transforms.ToTensor(),        # 将数据转为张量
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                        std = [0.229, 0.224, 0.225])
                                ]), download = True)
cifar_testset = DataLoader(cifar_testset, 
                            batch_size = BATCH_SIZE, 
                            shuffle = True)    


'''
预览数据集
'''
def  previewCifarLoader(p_cifar_trainset, p_cifar_testset):
    iterator = iter(p_cifar_trainset)
    try:
        x, label = next(iterator)
        print("训练集张量：{0},标签张量：{1}".format(x.shape, label.shape))
    except StopIteration:
        print("迭代器已耗尽,无元素可返回")
        return None
    iterator = iter(p_cifar_testset)
    try:
        x, label = next(iterator)
        print("测试集张量：{0},标签张量：{1}".format(x.shape, label.shape))
    except StopIteration:
        print("迭代器已耗尽,无元素可返回")
        return None
    return 0


def run():
    '''
    实例化网络对象
    '''
    model = ResNet18()

    '''
    实例化交叉熵损失函数对象
    '''
    criteon = nn.CrossEntropyLoss()

    '''
    实例化优化器
    '''
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)


    '''
    训练网络
    '''
    for epoch in range(EPOCHS):
        # 批次  (图片, 标签)
        for batch_idx, (x, label) in enumerate(cifar_trainset):
            logits = model(x) # 预测结果
            #计算交叉熵损失函数
            loss = criteon(logits, label)
            
            #参数迭代
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss:{0}".format(loss.item()))
        '''
        测试验证
        '''
        total_correct = 0
        total_num = 0
        for x, label in cifar_testset:
            logits = model(x)
            pred = logits.argmax(dim = 1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
        print("第 {0} 轮训练结束,当前目标函数值为：{1} ,模型正确率为：{2}%".format(epoch + 1, loss.item(), (total_correct / total_num) * 100))



    return 0

if __name__ == '__main__':
    run()

