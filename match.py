#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

nkernels = 100
overlap_feats_dim = 4
#hidden_dim = 512
n_outs = 2
dropout_rate = 0.5

class Match(nn.Module):
    def __init__(self):
        super(Match, self).__init__()
        self.sim_mat = nn.Parameter(torch.Tensor(nkernels, nkernels))
        self.fc1 = nn.Linear(nkernels * 2 + 1 + overlap_feats_dim, nkernels * 2 + 1 + overlap_feats_dim)
        self.fc2 = nn.Linear(nkernels * 2 + 1 + overlap_feats_dim, n_outs)
        #self.fc1 = nn.Linear(nkernels * 2 + 1 + overlap_feats_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, n_outs)
        init.xavier_uniform(self.sim_mat.data)
        init.xavier_uniform(self.fc1.weight.data)
        init.xavier_uniform(self.fc2.weight.data)

    def forward(self, q_feats, a_feats, overlap_feats):
        sim = torch.bmm(torch.mm(q_feats, self.sim_mat).unsqueeze(1), a_feats.unsqueeze(2)).squeeze(2)
        flatten = torch.cat((q_feats, sim, a_feats, overlap_feats), 1)
        hidden = F.dropout(F.tanh(self.fc1(flatten)), dropout_rate)
        #hidden = F.dropout(F.relu(self.fc1(flatten)), dropout_rate)
        prob = self.fc2(hidden)
        return prob

if __name__ == '__main__':
    import numpy as np
    from dataset import Dataset
    from sentence import Sentence
    from torch.utils import data
    sentence = Sentence()
    train_data = np.load('./data/train/data.npz')
    dev_data = np.load('./data/dev/data.npz')
    #train_loader = DataLoader(train_data, train=True, shuffle=True, batch_size=50)
    dev_loader = data.DataLoader(Dataset(dev_data))
    #train_data = train_loader.next()
    dev_data = iter(dev_loader).next()
    #print(train_data[1].size())
    #q_feats, a_feats = sentence(Variable(train_data[0]), Variable(train_data[1]))
    q_feats, a_feats = sentence(Variable(dev_data[1]), Variable(dev_data[2]))
    print(q_feats.size(), a_feats.size())
    match = Match()
    #prob = match(q_feats, a_feats, Variable(train_data[2]))
    prob = match(q_feats, a_feats, Variable(dev_data[3]))
    print(prob.size())
