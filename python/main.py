############################################################
# File name: main.py
# Author: Abdelrahman Eldesokey
##    Email: abdelrahman.eldesokey@liu.se
# Date created: 2018-08-28
# Date last modified: 2018-08-28
# Python Version: 3.6
# Description: TSBB17 course project (1) starter code.
############################################################

import os
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

#from models.cvlNet import cvlNet
from models.goalNet import goalNet
from models.goalNetRGBHSV import goalNetRGBHSV
from train import train
from test import test


# Check if CUDA support is available (GPU)
use_cuda = torch.cuda.is_available()


##################################
## Download the CIFAR10 dataset ##
##################################


# Image transformations to apply to all images in the dataset (Data Augmentation)
transform_train = transforms.Compose([
    # Convert images to Tensors (The data structure that is used by Pytorch)
    transforms.ToTensor(),
    # Normalize the images to zero mean and unit std
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Image transformations for the test set.
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Specify the path to the CIFAR-10 dataset and create a dataloader where you specify the "batch_size"
trainset = torchvision.datasets.CIFAR10(
    root='/courses/TSBB17', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/courses/TSBB17', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

# Specify classes labels
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


####################################
## Init the network and optimizer ##
####################################

# Load and initialize the network architecture
#model = cvlNet()
#model = goalNet()
model = goalNetRGBHSV()

if use_cuda:
    model.cuda()
    cudnn.benchmark = True

# The objective (loss) function
objective = nn.NLLLoss()

# The optimizer used for training the model
optimizer = torch.optim.Adam(model.parameters())


#######################
## Train the network ##
#######################

start_epoch = 1
num_epochs = 50
model, loss_log, acc_log = train(
    model, trainloader, optimizer, objective, use_cuda, start_epoch, num_epochs=num_epochs)


##########################
## Evaluate the network ##
##########################
test_acc = test(model, testloader, use_cuda)
