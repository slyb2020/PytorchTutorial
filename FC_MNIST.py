import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from FC_Net import FC_Net
from MNIST_Dataset import *
import time
import tqdm

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        Print("Using CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fcNetMNIST = FC_Net(hidden=[512, 512, 512])
    optimizer = torch.optim.SGD(fcNetMNIST.parameters(), lr=1e-3, momentum=0.9)
    Loss = nn.CrossEntropyLoss()
    maxEpoch = 60
    for epoch in range(maxEpoch):
        fcNetMNIST.train()
        lossEpoch = 0
        startTime = time.time()
        for images, labels in trainLoader:
            optimizer.zero_grad()
            prediction = fcNetMNIST(images)
            loss = Loss(prediction, labels)
            loss.backward()
            lossEpoch += loss
            optimizer.step()
        trainTime = time.time()

        accuracyTotal = 0
        fcNetMNIST.eval()
        with torch.no_grad():
            for images, labels in testDataset:
                prediction = fcNetMNIST(images)
                accuracy = (torch.argmax(prediction,dim=1)==labels).sum()
                accuracyTotal += accuracy
            testTime = time.time()
        print("完成{}/{}代训练，  损失：{}，   当前准确率：{}/{}={}，  训练用时：{}，    测试用时：{}".format(epoch+1, maxEpoch,
         lossEpoch, accuracyTotal, testSize, 100 * accuracyTotal/testSize, trainTime - startTime, testTime - trainTime))
