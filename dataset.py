#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

class Dataset:
    def __init__(self, datas, test=False):#, train=False):
        #self.train = train
        self.test = test
        self.qids = datas['qids']
        self.questions = datas['questions']
        self.answers = datas['answers']
        self.overlap_feats = datas['overlap_feats']
        if not self.test:
            self.labels = datas['labels']
        self.q_overlap_indices = datas['q_overlap_indices']
        self.a_overlap_indices = datas['a_overlap_indices']
        vocab_embeddings = datas['vocab_embeddings']
        #if not self.train:
        #    self.unique_qids = list(set(self.qids))
        self.embedding = nn.Embedding(vocab_embeddings.shape[0], vocab_embeddings.shape[1])
        self.embedding.weight.data.copy_(torch.Tensor(vocab_embeddings))

    def __getitem__(self, idx):
        #if self.train:
        #    return self.embedding(Variable(torch.from_numpy(self.questions[idx]).long())).data, \
        #           self.embedding(Variable(torch.from_numpy(self.answers[idx]).long())).data, \
        #           torch.from_numpy(self.overlap_feats[idx]), \
        #           torch.from_numpy(np.array([self.labels[idx]]))[0]
        #else:
        #    idx = self.qids == self.unique_qids[idx]
        #    return self.embedding(Variable(torch.from_numpy(self.questions[idx]).long())).data, \
        #           self.embedding(Variable(torch.from_numpy(self.answers[idx]).long())).data, \
        #           torch.from_numpy(self.overlap_feats[idx]), \
        #           torch.from_numpy(self.labels[idx])
        if self.test:
            return torch.from_numpy(np.array([self.qids[idx]]))[0], \
               torch.cat((self.embedding(Variable(torch.from_numpy(self.questions[idx]).long())).data, torch.from_numpy(self.q_overlap_indices[idx]).unsqueeze(1).float()), 1),\
               torch.cat((self.embedding(Variable(torch.from_numpy(self.answers[idx]).long())).data, torch.from_numpy(self.a_overlap_indices[idx]).unsqueeze(1).float()), 1), \
               torch.from_numpy(self.overlap_feats[idx])

        return torch.from_numpy(np.array([self.qids[idx]]))[0], \
               torch.cat((self.embedding(Variable(torch.from_numpy(self.questions[idx]).long())).data, torch.from_numpy(self.q_overlap_indices[idx]).unsqueeze(1).float()), 1),\
               torch.cat((self.embedding(Variable(torch.from_numpy(self.answers[idx]).long())).data, torch.from_numpy(self.a_overlap_indices[idx]).unsqueeze(1).float()), 1), \
               torch.from_numpy(self.overlap_feats[idx]), \
               torch.from_numpy(np.array([self.labels[idx]]))[0]

    def __len__(self):
        #if self.train:
        #    return self.qids.shape[0]
        #return len(self.unique_qids)
        return self.qids.shape[0]

if __name__ == '__main__':
    train_data = np.load('./data/train/data.npz')
    #dev_data = np.load('./data/dev/data.npz')
    train_dataset = Dataset(train_data)#, train=True)
    #dev_dataset = Dataset(dev_data)#, train=False)
    train_loader = data.DataLoader(train_dataset, shuffle=True)#, batch_size=50)
    #dev_loader = data.DataLoader(dev_dataset)#, shuffle=True)
    for i, data in enumerate(train_loader):
        print(data[1].size(), data[2].size)
    #train_data = iter(train_loader).next()
    #print(train_data[1].size(), train_data[2].size(), train_data[3].size(), train_data[4].size())
    #dev_data = iter(dev_loader).next()
    #print(dev_data[1].size(), dev_data[2].size(), dev_data[3].size(), dev_data[4].size())
