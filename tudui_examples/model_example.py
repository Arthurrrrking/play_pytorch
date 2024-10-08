import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten


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

if __name__=="__main__":
    zz = ZZ()
    input = torch.ones(64, 3, 32, 32)
    output = zz(input)
    print(output.shape)

