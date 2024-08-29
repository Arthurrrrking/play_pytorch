import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class ZZ(nn.Module):
    def __init__(self  ):
        super().__init__()
        self.relu1 = ReLU()
        self.segmoid1 = Sigmoid()

    def forward(self, input):
        output = self.segmoid1(input)
        return output

zz = ZZ()
writer = SummaryWriter("../logs_liner")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = zz(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()




