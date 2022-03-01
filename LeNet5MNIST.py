import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from LeNet import LeNet5
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainDataset = MNIST(root="D:\\WorkSpace\\DataSet", train=True, transform=torchvision.transforms.ToTensor())
testDataset = MNIST(root="D:\\WorkSpace\\DataSet", train=False, transform=torchvision.transforms.ToTensor())
batchSize = 100
trainSize = trainDataset.__len__()
testSize = testDataset.__len__()
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size = batchSize, shuffle=False)


leNet5 = LeNet5()
leNet5.to(device)
optimizer = torch.optim.SGD(leNet5.parameters(), lr=1e-3, momentum=0.9)
Loss = torch.nn.CrossEntropyLoss()
Loss.to(device)
maxEpoch = 20
for epoch in range(maxEpoch):
    leNet5.train()
    lossEpoch = 0
    startTime = time.time()
    for imgs, labels in trainLoader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predict = leNet5(imgs)
        loss = Loss(predict, labels)
        loss.backward()
        optimizer.step()
        lossEpoch += loss
    endTime = time.time()
    leNet5.eval()
    with torch.no_grad():
        accuracyTotal = 0
        for imgs, labels in testLoader:
            imgs = imgs.to(device)
            labels= labels.to(device)
            predict = leNet5(imgs)
            accuracy = (torch.argmax(predict, 1)==labels).sum()
            accuracyTotal+=accuracy
    testTime = time.time()
    print("已训练{}/{}代， 损失：{}，准确率：{}， 训练用时{}，测试用时{}".format(epoch+1, maxEpoch, lossEpoch,
                                                          100*accuracyTotal/testSize, endTime-startTime, testTime - endTime))


