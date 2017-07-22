#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from sentence import Sentence
from match import Match

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sentence = Sentence()
        self.match = Match()

    def forward(self, q_input, a_input, overlap_feats):
        q_feats, a_feats = self.sentence(q_input, a_input)
        prob = self.match(q_feats, a_feats, overlap_feats)
        return prob

if __name__ == '__main__':
    import numpy as np
    from dataloader import DataLoader
    from torch.autograd import Variable
    train_data = np.load('./data/train/data.npz')
    dev_data = np.load('./data/dev/data.npz')
    train_loader = DataLoader(train_data, train=True, shuffle=True, batch_size=50)
    dev_loader = DataLoader(dev_data, train=False, shuffle=True, batch_size=1)
    train_data = train_loader.next()
    dev_data = dev_loader.next()
    net = Net()
    train_prob = net(Variable(train_data[0]), Variable(train_data[1]), Variable(train_data[2]))
    dev_prob = net(Variable(dev_data[0].squeeze(0)), Variable(dev_data[1].squeeze(0)), Variable(dev_data[2].squeeze(0)))
    print train_prob.size()
    print dev_prob.size()
