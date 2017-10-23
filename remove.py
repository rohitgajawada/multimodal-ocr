from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset_online as dataset
import pickle
from collections import OrderedDict
import operator

workers = 1
trainroot = '/home/sreekar/Desktop/words/train/'
train_dataset = dataset.lmdbDataset(root=trainroot)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32,
    shuffle=True, sampler=None,
    num_workers=int(workers),
    collate_fn=dataset.alignCollate(imgH=32, imgW=512, keep_ratio=False))

val_iter = iter(train_loader)
max_iter = 1000
i = 0
n_correct = 0
loss_avg = utils.averager()

max_iter = 2
for i in range(max_iter):
    data = val_iter.next()
    i += 1
    cpu_images, cpu_stk, cpu_texts = data
