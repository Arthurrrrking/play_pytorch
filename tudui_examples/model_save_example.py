# 模型的保存
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16()

# 保存方式1(保存网络模型的结构+模型参数)
torch.save(vgg16, "vgg16_save_test.pth")

# 保存方式2（仅保存模型的参数，为字典格式。官方推荐）
torch.save(vgg16.state_dict(), "vgg16_save_test2.pth")


# 陷阱
class ZZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5)

    def forward(self, input):
        return self.conv1(input)

zz = ZZ()
torch.save(zz, "zz.pth")