import torch
import torch.nn as nn
from torch.autograd import Variable



def test(model, testloader, use_cuda):

    correct = 0
    total = 0
    for idx, (inputs, targets) in enumerate(testloader):

        # Move data to GPU if CUDA is available
        if use_cuda: 
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs = torch.tensor(inputs, requires_grad=False)

        # Feed-forward the network
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    return 100 * correct / total
