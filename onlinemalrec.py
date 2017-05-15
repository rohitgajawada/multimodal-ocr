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

#images are of variable size
#make parameter size code
arr = [str(x) for x in range(1, 130)]

strokes = open("1.stk")
cntr = 0
x_ar = []
y_ar = []
for line in strokes:
    if cntr<2:
        cntr += 1
        continue
    parsed = line.split(' ')
    x_ar.append(int(parsed[0]))
    y_ar.append(int(parsed[1]))

x_ar = np.array(x_ar)
y_ar = np.array(y_ar)
x_ar = (x_ar - min(x_ar))*100.0/(max(x_ar)-min(x_ar))
y_ar = (y_ar - min(y_ar))*100.0/(max(y_ar)-min(y_ar))

z = np.vstack((x_ar, y_ar))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 2)
        self.conv2 = nn.Conv2d(3, 3, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
output = net(z)
print(output)
