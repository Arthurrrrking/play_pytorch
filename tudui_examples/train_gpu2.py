"""
使用GPU进行模型训练，与train.py代码中不同的是，需要在下面：
网络模型
数据（输入，标注(即target)）
损失函数
先定义一个divice对象，指向使用cpu还是gpu，然后在上面三个对象中使用.to()转一道，即完成使用指定硬件进行训练
如：
zz = ZZ()
zz = zz.cuda()  # 表示使用GPU进行训练
"""

import math

import torch
from torch import nn
import torchvision.datasets
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

# device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")  # device("cuda:0")方法等价于device("cuda")，但是device("cuda:1")表示使用第二块cpu进行训练，更灵活

divice = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 这条语句等价于上面三行

# 准备数据集
train_dataset = torchvision.datasets.CIFAR10("../data", train=True, transform=ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=ToTensor(), download=True)

# 数据集长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print(f"train_dataset_size: {train_dataset_size}, test_dataset_size: {test_dataset_size}")

# 利用Dataload加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 创建神经网络
class ZZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        return self.model1(input)
zz = ZZ()
zz.to(device)

# 损失函数
loss_fn = CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2  # 1e-2 = 0.01
optimizer = torch.optim.SGD(zz.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 5

writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print(f"--------------第{i+1}轮训练开始----------------")
    # 训练步骤开始
    zz.train()  # 这一句代码非必须，因为train()方法只对部分层生效，比如Dropout层, BatchNorm层
    for data in train_dataloader:
        imgs, target = data
        imgs = imgs.to(device)
        target = target.to(device)
        output = zz(imgs)
        loss = loss_fn(output, target)

        # 优化器优化模型。注意：在每一次通过反向传播传递最新的梯度信息的时候要先把上一次的梯度清零
        optimizer.zero_grad()  # 在反向传播之前，将所有模型参数的梯度缓存清零，以防止梯度累积。不清零梯度会累加。
        loss.backward()  # 反向传播，计算损失函数相对于模型参数的梯度，并将其存储在每个参数的 grad 属性中。
        optimizer.step()  # 优化器根据刚刚计算的梯度来更新模型的参数（权重）

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}/{math.ceil(train_dataset_size/64)}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        zz.eval()   # 这一句代码非必须，因为eval()方法只对部分层生效，比如Dropout层, BatchNorm层
        for data in test_dataloader:
            imgs, target = data
            imgs = imgs.to(device)
            target = target.to(device)
            output = zz(imgs)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            accuracy_num = (output.argmax(1) == target).sum().item()
            total_accuracy += accuracy_num

        print(f"第{i+1}轮训练完成，总的Loss: {total_loss}")
        print(f"第{i+1}轮测试真确率：{total_accuracy}/{test_dataset_size}")
        writer.add_scalar("test_loss_total", total_loss, i)
        writer.add_scalar("test_accuracy", round(total_accuracy/test_dataset_size), i)

    torch.save(zz, f"zz_epoch_{i+1}.pth")
    print(f"第{i+1}轮训练模型已保存")

writer.close()








