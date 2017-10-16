# from __future__ import print_function
# import argparse
# import random
# import torch
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data
# from torch.autograd import Variable
# import numpy as np
# from warpctc_pytorch import CTCLoss
# import os
# import utils
# import dataset
# import pickle
# from collections import OrderedDict
# import operator
# import torch.nn as nn
#
# import models.crnn as crnn
#
#
#
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--trainroot', required=True, help='path to dataset')
# parser.add_argument('--valroot', required=True, help='path to dataset')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
# parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
# parser.add_argument('--imgW', type=int, default=256, help='the width of the input image to network')
# parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
# parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, default=0.00005')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
# # parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
# parser.add_argument('--experiment', default=None, help='Where to store samples and models')
# parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
# parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
# parser.add_argument('--valInterval', type=int, default=1400, help='Interval to be displayed')
# parser.add_argument('--saveInterval', type=int, default=1500, help='Interval to be displayed')
# parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
# parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
# parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
# parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
# opt = parser.parse_args()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# class BidirectionalLSTM(nn.Module):
#
#     def __init__(self, nIn, nHidden, nOut):
#         super(BidirectionalLSTM, self).__init__()
#
#         self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
#         self.embedding = nn.Linear(nHidden * 2, nOut)
#
#     def forward(self, input):
#         recurrent, _ = self.rnn(input)
#         T, b, h = recurrent.size()
#         t_rec = recurrent.view(T * b, h)
#
#         output = self.embedding(t_rec)  # [T * b, nOut]
#         output = output.view(T, b, -1)
#
#         return output
#
# nh = 256
# model_path = './data/crnn.pth'
# alphabet = '0123456789abcdefghijklmnopqrstuvwxyz!.,\'"#()&+*-/;:?'
# nclass = len(alphabet) + 1
# crnn = crnn.CRNN(32, 1, 37, 256)
# if torch.cuda.is_available():
#     crnn = crnn.cuda()
#
# crnn.load_state_dict(torch.load(model_path))
# crnn.rnn = nn.Sequential(*list(crnn.rnn.children())[:-1])
# crnn.rnn.add_module('{0}'.format(1),BidirectionalLSTM(nh, nh, nclass))
#
#
#
# train_dataset = dataset.lmdbDataset(root=opt.trainroot)
# sampler = None
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=opt.batchSize,
#     shuffle=True, sampler=sampler,
#     num_workers=int(opt.workers),
#     collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
# test_dataset = dataset.lmdbDataset(
#     root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))
#
# converter = utils.strLabelConverter(alphabet)
# criterion = CTCLoss()
#
#
# image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
# text = torch.IntTensor(opt.batchSize * 5)
# length = torch.IntTensor(opt.batchSize)
#
# if opt.cuda:
#     crnn.cuda()
#     crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
#     image = image.cuda()
#     criterion = criterion.cuda()
#
# # print (crnn)
#
# image = Variable(image)
# text = Variable(text)
# length = Variable(length)
#
# # loss averager
# loss_avg = utils.averager()
#
# # setup optimizer
# if opt.adam:
#     optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
#                            betas=(opt.beta1, 0.999))
# elif opt.adadelta:
#     optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
# else:
#     optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)
#
#
# def val(net, dataset, criterion, max_iter=100):
#     print('Start val')
#
#     for p in crnn.parameters():
#         p.requires_grad = False
#
#     net.eval()
#     data_loader = torch.utils.data.DataLoader(
#         dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
#     val_iter = iter(data_loader)
#
#     i = 0
#     n_correct = 0
#     loss_avg = utils.averager()
#
#     max_iter = min(max_iter, len(data_loader))
#     for i in range(max_iter):
#         data = val_iter.next()
#         i += 1
#         cpu_images, cpu_texts = data
#         batch_size = cpu_images.size(0)
#         utils.loadData(image, cpu_images)
#         t, l = converter.encode(cpu_texts)
#         utils.loadData(text, t)
#         utils.loadData(length, l)
#
#         preds = crnn(image)
#         preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
#         cost = criterion(preds, text, preds_size, length) / batch_size
#         loss_avg.add(cost)
#
#         _, preds = preds.max(2)
#         preds = preds.squeeze(2)
#         preds = preds.transpose(1, 0).contiguous().view(-1)
#         sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
#         for pred, target in zip(sim_preds, cpu_texts):
#             if pred == target.lower():
#                 n_correct += 1
#
#     raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
#     for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
#         print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
#
#     accuracy = n_correct / float(max_iter * opt.batchSize)
#     print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
#
#
# def trainBatch(net, criterion, optimizer):
#     data = train_iter.next()
#     cpu_images, cpu_texts = data
#     batch_size = cpu_images.size(0)
#     utils.loadData(image, cpu_images)
#     t, l = converter.encode(cpu_texts)
#     utils.loadData(text, t)
#     utils.loadData(length, l)
#
#     preds = crnn(image)
#     preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
#     cost = criterion(preds, text, preds_size, length) / batch_size
#     crnn.zero_grad()
#     cost.backward()
#     optimizer.step()
#     return cost
#
#
# for epoch in range(opt.niter):
#     train_iter = iter(train_loader)
#     i = 0
#     while i < len(train_loader):
#         for p in crnn.parameters():
#             p.requires_grad = True
#         crnn.train()
#
#         cost = trainBatch(crnn, criterion, optimizer)
#         loss_avg.add(cost)
#         i += 1
#
#         if i % opt.displayInterval == 0:
#             print('[%d/%d][%d/%d] Loss: %f' %
#                   (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
#             loss_avg.reset()
#
#         if i % opt.valInterval == 0:
#             val(crnn, test_dataset, criterion)
#
#         # do checkpointing
#         if i % opt.saveInterval == 0:
#             torch.save(
#                 crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))





def decode(t, length, raw=False):
    """Decode encoded texts back into strs.

    Args:
        torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
        torch.IntTensor [n]: length of each text.

    Raises:
        AssertionError: when the texts and its length does not match.

    Returns:
        text (str or list of str): texts to convert.
    """
    # if length.numel() == 1:
    #     length = length[0]
    #     assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
    #     if raw:
    #         return ''.join([self.alphabet[i - 1] for i in t])
    #     else:
    #         char_list = []
    #         for i in range(length):
    #             if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
    #                 char_list.append(self.alphabet[t[i] - 1])
    #         return ''.join(char_list)
    # else:
    #     # batch mode
    #     assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
    #     texts = []
    #     index = 0
    #     for i in range(length.numel()):
    #         l = length[i]
    #         texts.append(
    #             self.decode(
    #                 t[index:index + l], torch.IntTensor([l]), raw=raw))
    #         index += l
    lis = []
    pntr = 0

    for i in length:
        # print (t[pntr:pntr+i])
        text = ''
        for j in range(i):
            if t[pntr+j] == 0:
                continue
            if text != '':
                val = text.split(',')[-2]
                # print text , val,t[pntr+j]
                if int(val) == t[pntr+j]:
                    continue
            text += str(t[pntr+j])+','
        print (text)
        pntr += i

        lis.append(text[:-1])
    return lis


ts = [1,16,13,15,14,15,15,15,0,0,1,1,1,4,4,5,6,6,6,7]
ls = [8,12]
decode(ts,ls)
