import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class goalNet(nn.Module):
    def __init__(self):

        super(goalNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4)

        # OLD
        # self.fc1 = nn.Conv2d(32, 32, 4) # Fully connected since expects a 4x4xc feature map

        self.prediction = nn.Linear(64, 10)
        self.loss = nn.LogSoftmax(1)

    def forward(self, x):
        # Green 1

        out = self.conv1(x)

        # Orange 1
        out = F.max_pool2d(out, kernel_size=(2, 2))

        # Red 1
        out = F.relu(out)  # relu1

        # Green 2
        out = self.conv2(out)

        # Red 2
        out = F.relu(out)  # relu2

        # Orange 2
        out = F.max_pool2d(out, kernel_size=(2, 2))

        # Green 3
        out = self.conv3(out)

        # Red 3
        out = F.relu(out)  # relu3

        # Orange 3
        out = F.max_pool2d(out, kernel_size=(2, 2))

        # Green 4
        out = self.conv4(out)

        # Red 4
        out = F.relu(out)  # relu4

        out = out.view(-1, 64)

        out = self.prediction(out)
        out = self.loss(out)

        return out
