#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import pickle


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

        # print self.dict

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        # print text
        t = []
        l = []
        for i in range(len(text)):
            t += pickle.loads(text[i])
            l.append(len(pickle.loads(text[i])))
        # print t
        # print l
        return (torch.IntTensor(t), torch.IntTensor(l))

    def unpickle(self,text):
        t = []
        for i in range(len(text)):
            t.append(pickle.loads(text[i]))
        return t

    def decode(self, t, length, raw=False):
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
        if raw == False:
            # print("length", length)
            for i in length:
                # print (t[pntr:pntr+i])
                text = ''
                for j in range(i):
                    if t[pntr+j] == 0:
                        continue
                    if text != '':
                        val = text.split(',')[-2]
                        if int(val) == t[pntr+j]:
                            continue
                    text += str(t[pntr+j])+','

                pntr += i
                lis.append(text[:-1])
        else:
            for i in length:
                text = ''
                for j in range(i):
                    text += str(t[pntr+j])+','
                pntr += i
                lis.append(text[:-1])
        return lis


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

if __name__ == '__main__':
    test = strLabelConverter('abcdefghijklmnopqrstuvwxyz')
    print test.encode([[1,2,3,4],[5,6,7,8]])
