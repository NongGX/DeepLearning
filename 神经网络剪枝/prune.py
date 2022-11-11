from copy import deepcopy

import cv2
import numpy as np
import torch
import torchvision.datasets
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Mynet

# 绘图包导入
import matplotlib

from prettytable import PrettyTable, MSWORD_FRIENDLY

matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

# 手动参数设置
lr = 0.01  # 学习率
epoch = 10  # 训练轮数
batch_size = 64  # batch_size
prec = 60  # 剪枝阈值百分位

# 数据集载入,使用FashionMNIST数据集
train_data = torchvision.datasets.FashionMNIST(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                               download=False)  # 下载训练集

test_data = torchvision.datasets.FashionMNIST(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                              download=False)  # 下载训练集

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)  # 装入Dataloader
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)  # 装入Dataloader

# 选择CUDA设备
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    print("GPU num:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print('Not using CUDA!!!')

# 创建模型
model = Mynet()
# print(model)  # 打印模型信息
if (use_cuda):
    model.to(device)  # 如果使用cuda将模型放到cuda中

# 优化器
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()


# 查看数据集图片
def show_img(index):
    i = 0
    for data in train_loader:
        imgs, targets = data
        if (i == 1):
            break
        else:
            print('输入图片大小:{}'.format(imgs[i].size()))
        i = i + 1

    if (index < batch_size):
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        print(len(images))
        img = images[index]
        img = img.reshape((28, 28)).numpy()
        img = img[:, :, [2, 1, 0]]  # BGR to RGB
        plt.imshow(img)
        plt.show()
    else:
        print("超出批次范围！")




def train(epochs,model):
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0

    min_loss = 100000

    for i in range(epochs):
        table = PrettyTable()
        print("-----第 {} 轮训练开始-----".format(i + 1))
        # 训练步骤开始
        model.train()  # 当网络中有dropout层、batchnorm层时，这些层能起作用
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            if use_cuda:
                data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = F.nll_loss(outputs, target)  # 计算实际输出与目标输出的差距

            # 优化器对模型调优
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播，计算损失函数的梯度
            optimizer.step()  # 根据梯度，对网络的参数进行调优

            total_train_step = total_train_step + 1

            if batch_idx % 10 == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {i+1} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


        # print("训练损失:{}".format(loss.item()))

        # --------测试步骤开始---------
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_loader:  # 测试数据集提取数据
                imgs, targets = data  # 数据放到cuda上
                if torch.cuda.is_available():
                    imgs = imgs.cuda()  # 数据放到cuda上
                    targets = targets.cuda()
                outputs = model(imgs)
                loss = F.nll_loss(outputs, targets)  # 仅data数据在网络模型上的损失
                total_test_loss = total_test_loss + loss.item()  # 所有loss调用item方法，减少cpu占用
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

            # print("整体测试集上的Loss：{}".format(total_test_loss))
            # print("整体测试集上的正确率：{}".format(total_accuracy / len(test_data)))
            total_test_step = total_test_step + 1

            if total_test_loss < min_loss:
                min_loss = total_test_loss
                print("\n保存模型,损失为:{}".format(total_test_loss/len(test_data)))
                torch.save(model.state_dict(), 'model.pth')

        table.field_names = ['训练损失', '测试平均损失', '测试平均精度']
        table.add_row([loss.item(), format(total_test_loss/len(test_data)), format(total_accuracy/len(test_data))])
        table.set_style(MSWORD_FRIENDLY)
        print(table)


# 前向推理预测
def predict(index):
    # 从测试集中取样
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    img = images[index]
    img = img.reshape((28, 28)).numpy()
    plt.imshow(img)
    plt.show()

    # 将测试图片转为一维的列向量
    img = torch.from_numpy(img)
    img = img.view(1, 784)
    img = img.to(device)

    # 载入训练好的权重
    model.load_state_dict(torch.load("model.pth"))

    # 进行前向推断，预测图片所在的类别
    with torch.no_grad():
        output = model.forward(img)
    ps = torch.exp(output)

    top_p, top_class = ps.topk(1, dim=1)
    labellist = ['T恤', '裤子', '套衫', '裙子', '外套', '凉鞋', '汗衫', '运动鞋', '包包', '靴子']
    prediction = labellist[top_class]
    probability = float(top_p)
    print(f'神经网络猜测图片里是 {prediction}，概率为{probability * 100}%')


# 百分位剪枝
def model_prune_by_percentile(model, prec):
    threshold_list = []
    for name, p in model.named_parameters():
        # 不删除偏差项
        if 'bias' in name:
            continue
        weight = p.data.cpu().numpy().flatten()  # 将权重参数拉平
        threshold = np.percentile(weight, prec)  # 根据阈值对权重参数进行筛选
        threshold_list.append(threshold)
        print(f'裁剪阈值:{threshold}')

    # 生成掩码
    masks = []
    index = 0
    for name, p in model.named_parameters():
        # 不删除偏差项
        if 'bias' in name:
            continue
        pruned_index = p.abs() > threshold_list[index]  # 返回布尔矩阵,利用了广播机制
        masks.append(pruned_index)
        index += 1

    return masks


# 打印权重信息
def print_weight(model):
    for p in model.parameters():
        print(p)


#  初始化训练
print("--- 初始化训练 ---")
train(epoch,model)
# print("--- 原始模型权重 ---")
# print_weight(model)


#  模型剪枝
print("--- 模型剪枝 ---")
masks = model_prune_by_percentile(model,60)
prune_model = deepcopy(model)
prune_model.prune(masks)
# print("--- 剪枝模型权重 ---")
# print_weight(model)

# 重新训练
print("--- 重新训练---")
model.load_state_dict(torch.load("model.pth")) #载入参数文件
train(epoch,model)
# print("--- 重新模型权重 ---")
# print_weight(model)
