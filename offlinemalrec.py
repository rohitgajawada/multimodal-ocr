import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import copy
import os

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

data_transforms = {
    'train': transform, 'val': transform
    }

data_dir = '/home/rohit/Documents/sreekarfiles/'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4, shuffle=True, num_workers=64) for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes
numclasses = len(dset_classes)

use_gpu = torch.cuda.is_available()

# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)
#
# inputs, classes = next(iter(dset_loaders['train']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[dset_classes[x] for x in classes])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding = 1)

        self.fc1 = nn.Linear(512*9, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 129)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = self.pool(F.relu(self.conv8(x)))

        x = x.view(-1, 512*9)
        # print(x.size())

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training = self.training)
        x = self.fc3(x)
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
optimizer = optim.Adam(net.parameters(), lr = 0.001)

# net.pool.register_forward_hook(printnorm)
# n.register_forward_hook(printnorm)
# net.fc1.register_forward_hook(printnorm)
# net.fc2.register_forward_hook(printnorm)

#Training
for epoch in range(1):
    running_loss = 0
    for i, data in enumerate(dset_loaders['train'], 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        # print(inputs.size())
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
for i, data in enumerate(dset_loaders['val'], 0):
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
