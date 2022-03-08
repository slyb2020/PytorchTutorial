import torch
import torchvision
from torchvision.datasets import EMNIST
# entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
# print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))

dataset = EMNIST("D:\\WorkSpace\\DataSet\\",split='mnist', download=True)
img, label = dataset[0]
print(img, label)
# torchvision.datasets.EMNIST("E:\\WorkSpace\\Dataset\\", split='mnist', download=True)
# model = torchvision.models.resnet18(pretrained=True)
# print(model)
# torchvision.datasets.EMNIST("E:\\WorkSpace\\Dataset\\", split='mnist', download=True)
# model = torchvision.models.alexnet(pretrained=True)
# print(model)
