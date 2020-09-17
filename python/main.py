############################################################
##    File name: main.py
##    Author: Abdelrahman Eldesokey
##    Email: abdelrahman.eldesokey@liu.se
##    Date created: 2018-08-28
##    Date last modified: 2018-08-28
##    Python Version: 3.6
##    Description: TSBB17 course project (1) starter code.
############################################################

import os
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models.cvlNet import cvlNet
from models.goalNet import goalNet
from train import train
from test import test

import time #Used to time the training (and test) phase(s) of the network

# Check if CUDA support is available (GPU)
use_cuda = torch.cuda.is_available()


########################################
## Learning param eval specifications ##
########################################

# Evaluating Learning rate:
# Change the learning rate and number of epochs
learning_rate = learning_rate = 0.001 
epochs = 50

# Evaluating the image size:
# Change the img_size (default = 32), also change the image_size_factor in goalNet.py
img_size = 32

# Evaluating the weight initialisation:
# In goalNet.py, pick between the initialisation methods and uncomment the "self.initialize_weights(init_method)" line
 
##################################
## Download the CIFAR10 dataset ##
##################################



# Image transformations to apply to all images in the dataset (Data Augmentation)
transform_train = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),                # Convert images to Tensors (The data structure that is used by Pytorch)
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalize the images to zero mean and unit std
])

# Image transformations for the test set.
transform_test = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Specify the path to the CIFAR-10 dataset and create a dataloader where you specify the "batch_size"
trainset = torchvision.datasets.CIFAR10(root='/courses/TSBB17', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/courses/TSBB17', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Specify classes labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

####################################
## Init the network and optimizer ##
####################################

# Load and initialize the network architecture 
#model = cvlNet()
model = goalNet()

if use_cuda:
    model.cuda()
    cudnn.benchmark = True

# The objective (loss) function
objective = nn.NLLLoss()

# The optimizer used for training the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 


#######################
## Train the network ##
#######################
train_time_0 = time.time()

start_epoch = 1
num_epochs = epochs
model, loss_log, acc_log = train(model, trainloader, optimizer, objective, use_cuda, start_epoch, num_epochs=num_epochs)

train_time_1 = time.time()

##########################
## Evaluate the network ##
##########################
test_time_0 = time.time()

test_acc = test(model, testloader, use_cuda)

test_time_1 = time.time()

#################################
## Display train and test time ##
#################################
print('#############################################')
print('Epochs: {0}, Learning rate: {1}'.format(num_epochs, learning_rate))
print('The training time is: {0}s, or {1}min'.format((train_time_1-train_time_0), (train_time_1-train_time_0)/60))
print('The test time is: {0}'.format(test_time_1-test_time_0))
print('#############################################')