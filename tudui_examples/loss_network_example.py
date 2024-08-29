# 损失函数
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class ZZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        return self.model1(input)


# imgs, target = next(iter(dataloader))
# print(target)
loss = CrossEntropyLoss()
zz = ZZ()
for imgs, target in dataloader:
    print(target)
    output = zz(imgs)
    loss_output = loss(output, target)
    loss_output.backward()  # 反向传播，梯度信息会传递给模型zz。模型中的每一层都会有一个权重信息，反向传播的梯度信息在每一层的权重信息内部
    print(loss_output)
    break



