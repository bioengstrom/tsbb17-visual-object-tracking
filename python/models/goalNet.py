import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class goalNet(nn.Module):
    def __init__(self):
        
        super(goalNet, self).__init__()
       
        img_size_factor = 1 

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4 * img_size_factor)

        self.prediction = nn.Linear(64, 10)
        self.loss = nn.LogSoftmax(1)

        # Different initialization methods:
        #init_method = nn.init.normal_
        #init_method = nn.init.uniform_
        #init_method = nn.init.xavier_normal_
        #init_method = nn.init.xavier_uniform_
        #init_method = nn.init.kaiming_normal_
        #init_method = nn.init.kaiming_uniform_
        #init_method = nn.init.orthogonal_
        #init_method = nn.init.sparse_ # Requires a 2D input tensor

        #self.initialize_weights(init_method)
    
    def forward(self, x):
        out = self.conv1(x) # Green 1
        out = F.max_pool2d(out, kernel_size=(2, 2))  # Orange 1
        out = F.relu(out)  # Red 1 # relu1
        out = self.conv2(out) # Green 2
        out = F.relu(out) # Red 2 # relu2
        out = F.max_pool2d(out, kernel_size=(2, 2)) # Orange 2
        out = self.conv3(out) # Green 3
        out = F.relu(out) # Red 3 # relu3
        out = F.max_pool2d(out, kernel_size=(2, 2)) # Orange 3
        out = self.conv4(out) # Green 4
        out = F.relu(out) # Red 4 # relu4
        out = out.view(-1, 64)
        out = self.prediction(out)
        out = self.loss(out)
        
        return out

    #Weight initialization (network initialization)
    def initialize_weights(self, init_method):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_method(m.weight)

                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)

