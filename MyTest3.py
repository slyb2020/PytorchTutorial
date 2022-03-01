import cv2
import torch
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10,CelebA,Caltech101
from torch.utils.data import DataLoader
from PIL import Image


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((30,30)),
    torchvision.transforms.Resize((400, 400)),
    torchvision.transforms.ToTensor(),
])
# myDataset = CIFAR10("D:/WorkSpace/DataSet/CIFAR10",transform=torchvision.transforms.ToTensor())
# myDataset = CelebA("D:/WorkSpace/DataSet/",split="all", transform=torchvision.transforms.ToTensor())
myDataset = Caltech101("D:/WorkSpace/DataSet/Caltech", transform=transform)
myLoader = DataLoader(dataset=myDataset, batch_size=1, shuffle=True)
for img, label in myLoader:
    print(img.shape,label)
    img = torch.squeeze(img,dim=0)
    img = torch.permute(img,(2,1,0))
    cv2.imshow("pic", img.numpy())
    cv2.waitKey(0)

# myIter = iter(myDataset)
# for img, label in myIter:
#     print(img.shape,label)
#     img = torch.permute(img,(2,1,0))
#     cv2.imshow("pic", img.numpy())
#     cv2.waitKey(0)
#
# for i in range(10):
#     img, label = myIter.__next__()
#     print(img.shape)
#     img = torch.permute(img,(2,1,0))
#     cv2.imshow("pic", img.numpy())
#     cv2.waitKey(0)

# for img, label in myIter:
#     print(img)
