import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16()
# vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_false)
train_data = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

# 对模型的修改
# 方式 1
# vgg16_false.add_module("add_linear", nn.Linear(1000, 10))
# print(vgg16_false)

# 方式 2-1 可以直接使用add_module将对应层进行替换
# vgg16_false.classifier.add_module("6", nn.Linear(4096, 10))
# print(vgg16_false)
# 方式 2-2通过索引指定进行替换
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

