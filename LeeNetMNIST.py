import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from LeNet import LeNet5
from LeeNet import LeeNet
import time
from MNIST_Dataset import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# leNet5 = LeNet5()
model = LeeNet()
model.to(device)
# optimizer = torch.optim.SGD(leNet5.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
Loss = torch.nn.CrossEntropyLoss()
Loss.to(device)
maxEpoch = 20
for epoch in range(maxEpoch):
    model.train()
    lossEpoch = 0
    startTime = time.time()
    for imgs, labels in trainLoader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predict = model(imgs)
        loss = Loss(predict, labels)
        loss.backward()
        optimizer.step()
        lossEpoch += loss
    endTime = time.time()
    model.eval()
    with torch.no_grad():
        accuracyTotal = 0
        for imgs, labels in testLoader:
            imgs = imgs.to(device)
            labels= labels.to(device)
            predict = model(imgs)
            accuracy = (torch.argmax(predict, 1)==labels).sum()
            accuracyTotal+=accuracy
    testTime = time.time()
    print("已训练{}/{}代， 损失：{}，准确率：{}， 训练用时{}，测试用时{}".format(epoch+1, maxEpoch, lossEpoch,
                                                          100*accuracyTotal/testSize, endTime-startTime, testTime - endTime))


