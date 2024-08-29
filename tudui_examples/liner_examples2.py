
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class ZZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner1 = Linear(196608, 10)

    def forward(self, input):
        output = self.liner1(input)
        return output


zz = ZZ()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    linear_output = zz(output)
    print(linear_output.shape)

