# 计算正确率
import torch

output = torch.tensor([[0.2, 0.5, 0.9],
                       [0.3, 0.4, 0.8]])

# argmax(1)表示横向求最大值，并返回最大值的下标，argmax(0)表示纵向求最大值，并返回最大值的下标
print(output.argmax(1))
print(output.argmax(0))
predict = output.argmax(1)
target = torch.tensor([1, 2])
print(len(target))
print((predict == target).sum().item())



