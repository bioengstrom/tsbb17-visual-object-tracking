import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class cvlNet(nn.Module):
    def __init__(self):
        
        super(cvlNet, self).__init__()
       
        # 3 => Number of input channels, the picture has three channels: [R, G, B]
        # 32 => Number of output channels, the convolution produces 32 output channels
        # 5 => Indicates the kernel size. A single number indicates a square kernel with a size of 5x5
        # padding=2 => The amount of zero-padding on both sides of the input.
        #               A 5x5 kernel placed at (0, 0) will have a overshoot of 2 in both the X and Y dimension
        # See: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md for convolution visualization
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        

        # Between each convolutional layer we perform a 2D max pooling over the input signal
        # Example: F.max_pool2d(out, kernel_size=(8,8), stride=8)
        #
        # out => input
        # kernel_size=(8,8) => The kernel_size which is the size of the window to take a max over
        # stride=8 => The stride of the window, the default value is kernel_size
        #
        # See: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

        # Sometiems we perform a RELU (REctified Linear Unit).
        # The .relu function applies the RELU element-wise
        # Outputs the same shape as input


        # 32 => Number of input channels, the previous layer outputs 32 channels
        # 32 => Number of output channels, the convolution produces 32 output channels which is equal to the input
        # 4 => Indicates the kernel size. A single number indicates a square kernel with a size of 4x4
        self.fc1 = nn.Conv2d(32, 32, 4) # Fully connected since expects a 4x4xc feature map


        self.prediction = nn.Linear(32, 10)
        self.loss = nn.LogSoftmax(1)
       
    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=(8,8), stride=8) # outputs 4x4xc featurmap
        out = F.relu(out) # relu1

        out = self.fc1(out) 
        out = F.relu(out) # relu2
        
        # The -1 indicates that the number of rows should be calculated from the number of columns.
        # Example: If out is a 64 values long vector out.view(-1, 32) would turn it into a [2, 32] matrix because ? * 32 = 64 => 64 / 32 = 2
        out = out.view(-1, 32)
        
        out = self.prediction(out)
        
        out = self.loss(out)
        
        return out
