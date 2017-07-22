#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

num_input_channels = 1
nkernels = 100
#q_max_length = 20
#a_max_length = 40
q_filter_width = 5
a_filter_width = 5
#filter_width = 5
embed_dim = 400

class Sentence(nn.Module):
    def __init__(self):
        super(Sentence, self).__init__()
        self.conv_q = nn.Conv2d(num_input_channels, nkernels, (q_filter_width, embed_dim + 1), padding=(q_filter_width-1, 0))
        self.conv_a = nn.Conv2d(num_input_channels, nkernels, (a_filter_width, embed_dim + 1), padding=(a_filter_width-1, 0))
        #self.conv = nn.Conv2d(num_input_channels, nkernels, (filter_width, embed_dim + 1), padding=((filter_width-1)/2, 0))
        self.atten_mat = nn.Parameter(torch.Tensor(nkernels, nkernels))
        init.xavier_uniform(self.conv_q.weight.data)
        init.xavier_uniform(self.conv_a.weight.data)
        #init.xavier_uniform(self.conv.weight.data)
        init.xavier_uniform(self.atten_mat.data)

    def forward(self, q_input, a_input):
        #Q = F.tanh(self.conv_q(q_input.unsqueeze(1)).squeeze(3))
        #A = F.tanh(self.conv_a(a_input.unsqueeze(1)).squeeze(3))
        Q = F.tanh(self.conv_q(q_input.unsqueeze(1)).squeeze(3))
        A = F.tanh(self.conv_a(a_input.unsqueeze(1)).squeeze(3))
        #Q = F.tanh(self.conv(q_input.unsqueeze(1)).squeeze(3))
        #A = F.tanh(self.conv(a_input.unsqueeze(1)).squeeze(3))
        #G = F.tanh(torch.bmm(torch.mm(Q.transpose(1, 2).contiguous().view(-1, nkernels), self.atten_mat).view(-1, q_max_length - q_filter_width + 1, nkernels), A))
        #G = F.tanh(torch.bmm(torch.mm(Q.transpose(1, 2).contiguous().view(-1, nkernels), self.atten_mat).view(1, -1, nkernels), A))
        G = F.tanh(torch.bmm(torch.mm(Q.transpose(1, 2).contiguous().view(-1, nkernels), self.atten_mat).view(1, -1, nkernels), A))
        #q_max = F.max_pool2d(G, (1, a_max_length - a_filter_width + 1))
        #a_max = F.max_pool2d(G, (q_max_length - q_filter_width + 1, 1)).squeeze(1).unsqueeze(2)
        q_max = F.softmax(torch.max(G, 2)[0].squeeze(2)).unsqueeze(2)
        a_max = F.softmax(torch.max(G, 1)[0].squeeze(1)).unsqueeze(2)
        #q_tilder = torch.mul(Q, F.softmax(q_max.unsqueeze(1).expand_as(Q)))
        #a_tilder = torch.mul(A, F.softmax(a_max.unsqueeze(1).expand_as(A)))
        q_feats = torch.bmm(Q, q_max).squeeze(2)
        a_feats = torch.bmm(A, a_max).squeeze(2)
        return q_feats, a_feats

if __name__ == '__main__':
    sentence = Sentence()
    import numpy as np
    from dataset import Dataset
    train_data = np.load('./data/train/data.npz')
    train_dataset = Dataset(train_data)#, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)#, batch_size=50)
    #dev_loader = DataLoader(dev_data, train=False, shuffle=True)
    train_data = iter(train_loader).next()
    q_feats, a_feats = sentence(Variable(train_data[1]), Variable(train_data[2]))
    print(q_feats.size(), a_feats.size())
