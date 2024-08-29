"""
torchvision中的transforms例子。用于处理图片张量化，及对张量数据的归一化，及裁剪处理
"""

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter("../logs")
image = Image.open("C:/Users/Athurrrr/Desktop/desktop.png")
print(image)

# ToTensor
toTensor = transforms.ToTensor()
img_tensor = toTensor(image)
writer.add_image("toTensor", img_tensor)

# 使用tensorboard查看 tensorboard --logdir D:\work_space\idea\play_pytorch\logs

# Normalize
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
"""
img_tensor[0][0][0]
"""
print(f'{img_tensor[0][0][0]},{img_tensor[1][0][0]},{img_tensor[2][0][0]}')
img_tensor_norm = normalize(img_tensor)
print(f'{img_tensor_norm[0][0][0]},{img_tensor_norm[1][0][0]},{img_tensor_norm[2][0][0]}')
writer.add_image("Normalize", img_tensor_norm)

# Resize
resize = transforms.Resize([128, 128])
img_tensor_resize = resize(img_tensor)
writer.add_image("Resize", img_tensor_resize, 0)

# Compose
resize2 = transforms.Resize(512)
compose = transforms.Compose([toTensor, resize2])
img_tensor_resize2 = compose(image)
writer.add_image("Resize", img_tensor_resize2, 1)

# RandomCrop
random_crop = transforms.RandomCrop(128)
compose2 = transforms.Compose([toTensor, random_crop])
for i in range(10):
    random_crop_img = compose2(image)
    writer.add_image("RandomCrop", random_crop_img, i)


writer.close()  # writer必须close()不然tensorboard中无法展示

