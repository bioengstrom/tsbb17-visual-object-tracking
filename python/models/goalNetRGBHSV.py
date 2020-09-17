import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

class goalNetRGBHSV(nn.Module):
    def __init__(self):

        super(goalNetRGBHSV, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4)

        # OLD
        # self.fc1 = nn.Conv2d(32, 32, 4) # Fully connected since expects a 4x4xc feature map

        self.prediction = nn.Linear(64, 10)
        self.loss = nn.LogSoftmax(1)

    def forward(self, x):

        xRGB = x
        xHSV = x.clone()
        for i in range(0, x.size()[0]):
            y = x[i].cpu().detach().numpy()
            #y = torchvision.transforms.ToPILImage()(x[i].cpu().detach())
            #y_hsv = y.convert(mode='HSV')

            y = np.swapaxes(y, 0, 1)
            y = np.swapaxes(y, 1, 2)
            y_hsv = cv2.cvtColor(y, cv2.COLOR_RGB2HSV)
            y_hsv = np.swapaxes(y_hsv, 1, 2)
            y_hsv = np.swapaxes(y_hsv, 0, 1)
            xHSV[i] = torch.from_numpy(y_hsv).to(xHSV)
            #xHSV[i] = torchvision.transforms.ToTensor()(y)


        #outRGB = self.model(xRGB)
        outHSV = self.model(xHSV)

        out = outHSV

        # Green 3
        out = self.conv3(out)

        # Red 3
        out = F.relu(out)  # relu3

        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = self.conv4(out)

        # Red 4
        out = F.relu(out)  # relu4

        #out = outHSV + outRGB
        #out = torch.cat((outRGB, outHSV), 1)
        #outRGB = outRGB.view(-1, 64)

        out = out.view(-1, 64)
        out = self.prediction(out)
        out = self.loss(out)

        return out


    def model(self, x):
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
        #out = self.conv3(out)

        # Red 3
        #out = F.relu(out)  # relu3

        # Orange 3
        #out = F.max_pool2d(out, kernel_size=(2, 2))

        # Green 4
        #out = self.conv4(out)

        # Red 4
        #out = F.relu(out)  # relu4

        return out

