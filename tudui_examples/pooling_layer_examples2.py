"""
池化层例子
"""
import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class ZZ(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=kernel_size, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

zz2 = ZZ(2)
zz3 = ZZ(3)

writer = SummaryWriter("../logs_maxpool")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = zz2(imgs)
    writer.add_images("output", output, step)
    output2 = zz3(imgs)
    writer.add_images("output2", output2, step)
    step = step + 1

writer.close()



