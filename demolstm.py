import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

bd = True

conv1a = nn.Conv1d(1, 3, 3)
conv1b = nn.Conv1d(1, 3, 3)
conv_2d = nn.Conv2d(3, 6, 2)
lstm = nn.LSTM(2, 128, bidirectional = bd)
lstm2 = nn.LSTM(128, 128, bidirectional = bd)
embedding = nn.Linear(128 * 2, 128)

inputs = Variable(torch.randn(100, 8, 2))

xinputs, yinputs = inputs.chunk(2, 2)
xinputs = xinputs.contiguous().view(8, 100, 1)
yinputs = yinputs.contiguous().view(8, 100, 1)
newx, newy = xinputs.view(8, 1, -1), yinputs.view(8, 1, -1)
newx ,newy= conv1a(newx), conv1b(newy)
# newx, newy = newx.view(8, 3, -1), newy.view(8, 3, -1)
# print(newx.size())
# newx, newy= conv1a(newx), conv1b(newy)
# newx, newy = newx.view(8, 3, -1), newy.view(8, 3, -1)
# newx, newy= conv1a(newx), conv1b(newy)

fininput = torch.cat((newx, newy), 2).view(8, 3, 98, 2)

a = conv_2d(fininput)
rnn_input = a.view(-1, 8, 6)

# inputs = torch.cat(inputs).view(len(inputs), 8, -1)

# out, _ = lstm(inputs)
# T, b, h = out.size()
#
# out_1 = out.view(T * b, h)
# out_t = embedding(out_1)
#
# out_2 = out_t.view(T, b, -1)
# finalout, _ = lstm2(out_2)
# final = finalout[-1]
#
# print(final)
