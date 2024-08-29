# 优化器
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
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

zz = ZZ()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(zz.parameters(), lr=0.01)
for epoc in range(20):
    total_loss = 0
    for data in dataloader:
        imgs, target = data
        outputs = zz(imgs)
        result_loss = loss(outputs, target)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        total_loss += result_loss.item()
    print(f"epoc {epoc}, total_loss: {total_loss}")


