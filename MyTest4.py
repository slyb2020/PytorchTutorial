import torch
import torchvision
# entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
# print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))

# torchvision.datasets.EMNIST("E:\\WorkSpace\\Dataset\\", split='mnist', download=True)
model = torchvision.models.resnet18(pretrained=True)
print(model)