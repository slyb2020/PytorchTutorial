import torchvision
from torch import nn
import torch


resnet = torchvision.models.resnet18(pretrained=True)
print(resnet)
# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.
for param in resnet.conv1.parameters():
    print("param1=",param[0,0,0])
    break
for param in resnet.fc.parameters():
    print("param2=",(param[0,:10]).detach().numpy())
    break

# Forward pass.
labels = torch.randint(0,100,size=(64,))
print("labels=", labels)
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)

Loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(resnet.parameters(), lr=1e-3, momentum=0.9)
for _ in range(10):
    optim.zero_grad()
    predicts = resnet(images)
    loss = Loss(predicts, labels)
    print("loss=", loss)
    loss.backward()
    optim.step()
for param in resnet.conv1.parameters():
    print("param1=",param[0,0,0])
    break
for param in resnet.fc.parameters():
    print("param2=",(param[0,:10]).detach().numpy())
    break
