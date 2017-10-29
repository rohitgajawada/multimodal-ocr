import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cPickle

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        print("Using super model")

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        #parallel net1
        self.relu = nn.ReLU()

        self.conv1a = nn.Conv2d(1, 128, kernel_size=(1, 5), stride=(1, 1))
        self.bn1a = nn.BatchNorm2d(128)
        self.mp1a = nn.MaxPool2d((1, 2), (1, 2))

        self.conv2a = nn.Conv2d(128, 256, kernel_size=(1, 5), stride=(1, 1))
        self.bn2a = nn.BatchNorm2d(256)
        self.mp2a = nn.MaxPool2d((1, 2), (1, 2))

        self.conv3a = nn.Conv2d(256, 384, kernel_size=(2, 3), stride=(1, 1))
        self.bn3a = nn.BatchNorm2d(384)
        self.conv4a = nn.Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1))
        self.bn4a = nn.BatchNorm2d(384)
        self.conv5a = nn.Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
        self.bn5a = nn.BatchNorm2d(128)
        ######

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(640, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input, stk1, n):

        conv = self.cnn(input)
        print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"

        stk1 = stk1.contiguous().transpose(1, 2)
        stk1 = stk1.contiguous().view(-1, 1, 2, n)

        print(stk1.size())
        x = self.relu(self.bn1a(self.conv1a(stk1)))
        print(x.size())
        x = self.mp1a(x)
        print(x.size())
        x = self.relu(self.bn2a(self.conv2a(x)))
        print(x.size())
        x = self.mp2a(x)
        print(x.size())

        x = self.relu(self.bn3a(self.conv3a(x)))
        print(x.size())
        x = self.relu(self.bn4a(self.conv4a(x)))
        print(x.size())
        x = self.relu(self.bn5a(self.conv5a(x)))
        print(x.size())

        conv = torch.cat((conv, x), dim=1)
        print(conv.size())

        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        print(output.size())

        return output


# path = '/home/rohit/Documents/cvitwork/ocrnew/testing/testword293_2.p'
# f = open(path, 'rb')
# stk = cPickle.load(f)
# f.close()
# stk = np.array(stk, dtype=float)
# stk = Variable(torch.from_numpy(stk)).float()
#
# n = 544
# stk = Variable(torch.randn(8, n, 2))
# im = Variable(torch.randn(8, 3, 32, 512))
# net = CRNN(32, 3, 10, 256)
#
# output = net(im, stk, n)
