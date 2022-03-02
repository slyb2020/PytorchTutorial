import torch
import torchvision
from torch import nn


class FC_Net(nn.Module):
    def __init__(self, hidden=[256, 128, 64]):
        super(FC_Net, self).__init__()
        self.inputLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden[0]),
            nn.ReLU()
        )
        self.hiddenLayer1 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU()
        )
        self.hiddenLayer2 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU()
        )
        self.outputLayer = nn.Sequential(
            nn.Linear(hidden[2], 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.hiddenLayer1(x)
        x = self.hiddenLayer2(x)
        x = self.outputLayer(x)
        return x


