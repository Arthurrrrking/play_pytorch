# 模型的加载
import torch
import torchvision
from model_save_example import *

# 方式一，和模型的保存方式一相对应
# vgg16 = torch.load("vgg16_save_test.pth")
# print(vgg16)

# 方式二，和模型保存的方式二相对应
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_save_test2.pth"))
# model = torch.load("vgg16_save_test2.pth")
print(vgg16)


# 陷阱 load代码所在的模块必须要能读取到模型对应的结构类，比如在model_save_example.py中定义的ZZ，否则会报找不到类异常
zz = torch.load("zz.pth")
print(zz)

