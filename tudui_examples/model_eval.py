import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor

from model_example import ZZ
import torchvision
from PIL import Image
device = torch.device("cpu")

writer = SummaryWriter("../logs_model_eval")
img_path = "../images/dog2.png"
img = Image.open(img_path)
if img.mode == "RGBA":  # PNG图片有4通道RGBA，这里判断如果图片是4通道的，则清除透明度通道
    img = img.convert("RGB")
totensor = torchvision.transforms.ToTensor()
img_tensor2 = totensor(img)
writer.add_image("model_eval_img_before", img_tensor2)
transforms = torchvision.transforms.Compose([Resize([32, 32]), ToTensor()])
img_tensor = transforms(img)
writer.add_image("model_eval_img", img_tensor)
writer.close()
img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))

model_path = "../models/zz_model.pth"
zz = torch.load(model_path, map_location=torch.device("cpu"))
zz.to(device)
zz.eval()
with torch.no_grad():  # 禁用梯度计算，可以减少内存和计算量，加快输出结果
    output = zz(img_tensor)
print(output)
print(output.argmax(1))


