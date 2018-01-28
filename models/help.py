import torch
from torch.autograd import Variable
im = torch.randn(1, 3, 32, 128)

length = 3

# for i in range(0, 10):
start = 0
piece = im.narrow(3, start, start + length)
print(piece.size())

start = 1
piece = im.narrow(3, start, start + length)
print(piece.size())
