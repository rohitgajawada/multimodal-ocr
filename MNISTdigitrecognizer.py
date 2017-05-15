#achieves 93.7% on 10 epochs

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return x

def printnorm(self, input, output):
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('')

net = Net()
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# net.conv1.register_forward_hook(printnorm)
# net.conv2.register_forward_hook(printnorm)
# net.fc1.register_forward_hook(printnorm)
# net.fc2.register_forward_hook(printnorm)

#Training
for epoch in range(1):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if i % 20 == 19:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print("Training done")

#Testing
correct = 0
total = 0
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    tlabels = labels.cuda()
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    outputs = net(inputs)
    pred = outputs.data.max(1)[1]
    x = (pred == tlabels)
    x = sum(x)
    x = x[0]
    correct += x
    total += len(tlabels)

    print("Correct:", correct)
    print("Total:", total)
    print("Ratio", correct/total)
