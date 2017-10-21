import torch
import torch.nn as nn

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

        self.relu = nn.ReLU()

        #parallel net1
        self.conv1a = nn.Conv2d(1, 6, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.bn1a = nn.BatchNorm2d(6)
        self.mp1a = nn.MaxPool2d((1, 3), (1, 2))
        self.conv2a = nn.Conv2d(6, 16, kernel_size=(2, 3), stride=(1, 1), padding=(0, 0))
        self.bn2a = nn.BatchNorm2d(16)
        self.mp2a = nn.MaxPool2d((1, 3), (1, 2))

        #parallel net2
        self.conv1b = nn.Conv2d(1, 6, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.bn1b = nn.BatchNorm2d(6)
        self.mp1b = nn.MaxPool2d((1, 3), (1, 2))
        self.conv2b = nn.Conv2d(6, 16, kernel_size=(2, 3), stride=(1, 1), padding=(0, 1))
        self.bn2b = nn.BatchNorm2d(16)
        self.mp2b = nn.MaxPool2d((1, 3), (1, 2))

        #main net
        self.conv0 = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1)
        self.mp0 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.mp3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv4 = nn.Conv2d(260, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.mp5 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv6 = nn.Conv2d(520, 520, kernel_size=2, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(520)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(520, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, x, stk1):

        # sx, sy, sz = torch.split(stk, 1, 1)
        # sx, sy, sz = sx.contiguous().float().view(1, 1, 1, -1), sy.contiguous().float().view(1, 1, 1, -1), sz.contiguous().float().view(1, 1, 1, -1)
        #
        # stk1 = torch.cat((sx, sy), dim=2)

        # sx, sy = torch.split(stk, 1, 2)
        # sx, sy = sx.contiguous().float().view(1, 1, 1, -1), sy.contiguous().float().view(1, 1, 1, -1)
        #
        # stk1 = torch.cat((sx, sy), dim=2)

        stk1 = stk1.contiguous().transpose(1, 2)
        stk1 = stk1.contiguous().view(-1, 1, 2, 524)

        xa = self.relu(self.bn1a(self.conv1a(stk1)))
        print xa.size()
        xa = self.mp1a(xa)
        print xa.size()
        xa = self.relu(self.bn2a(self.conv2a(xa)))
        print xa.size()
        xa = self.mp2a(xa)
        print xa.size()

        print "-----------"

        xb = self.relu(self.bn1b(self.conv1b(stk1)))
        print xb.size()
        xb = self.mp1b(xb)
        print xb.size()
        xb = self.relu(self.bn2b(self.conv2b(xb)))
        print xb.size()
        xb = self.mp2b(xb)
        print xb.size()

        print "------------"

        xa = xa.view(-1, 4, 4, 129)
        xb = xb.view(-1, 8, 2, 130)

        print xa.size()
        print xb.size()

        x = self.relu(self.conv0(x))
        x = self.mp0(x)
        print(x.size())
        x = self.relu(self.conv1(x))
        x = self.mp1(x)
        print(x.size())
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.mp3(x)
        x = torch.cat((x, xa), dim=1)
        print(x.size())

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.mp5(x)
        x = torch.cat((x, xb), dim=1)
        print(x.size())

        x = self.relu(self.bn6(self.conv6(x)))
        print(x.size())
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)

        output = self.rnn(x)

        return output


# from torch.autograd import Variable
# import numpy as np
# import cPickle
#
# path = '/home/rohit/Documents/cvitwork/ocrnew/testing/testword293_2.p'
# f = open(path, 'rb')
# stk = cPickle.load(f)
# f.close()
# stk = np.array(stk, dtype=float)
#
# stk = Variable(torch.from_numpy(stk)).float()
#
# stk = Variable(torch.randn(8, 524, 2))
# im = Variable(torch.randn(8, 3, 32, 512))
# net = CRNN(32, 3, 10, 256)
#
# output = net(im, stk)
