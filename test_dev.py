#!/usr/bin/env python 
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import logging
import os.path
import os
import sys
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
# import torch.cuda as cuda
from torch.utils import data
from tensorboard_logger import configure, log_value

from dataset import Dataset
from net import Net

def train():
    prob_list =[]
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    n_epochs = 100
    batch_size = 50
    train_interval = 1000
    #test_interval = 500
    #test_steps = 100

    # cuda.set_device(1)
    configure("logs", flush_secs=5)


    # class_sample_count = [1, 1]
    # weight_per_class = 1 / torch.Tensor(class_sample_count).double()
    # train_data = np.load('./data/train/data.npz')
    dev_data = np.load('./data/dev/data.npz')

    # train_dataset = Dataset(train_data)#, train=True)
    dev_dataset = Dataset(dev_data)#, train=False)
    # print(len(dev_dataset))
    # weights = [weight_per_class[label] for label in train_data['labels']]
    # sampler = data.sampler.WeightedRandomSampler(weights, len(weights))
    # train_dataloader = data.DataLoader(train_dataset, sampler=sampler)#, batch_size=batch_size, sampler=sampler)
    dev_dataloader = data.DataLoader(dev_dataset)#, shuffle=True)

    net = Net()
    # criterion = nn.CrossEntropyLoss()
    # ignored_params = list(map(id, net.sentence.conv_q.parameters())) + list(map(id, net.sentence.conv_a.parameters()))
    #ignored_params = list(map(id, net.sentence.conv.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    # optimizer = optim.Adam([
    #     {'params': net.sentence.conv_q.parameters()},
    #     {'params': net.sentence.conv_a.parameters()},
    #     #{'params': net.sentence.conv.parameters()},
    #     {'params': base_params}
    #     ], lr=0.000003)

    latest_epoch_num = 5
    model_path = './model/epoch_6_2017-06-03#12:12:00.params'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        logger.info('Successfully loaded model: %s' % (model_path))
    else:
        logger.info('Could not find model: %s' % (model_path))

    MRR_file = open('MRR.txt', 'w')

    net.train()
    epoch=1
    running_loss = 0.0
    correct = 0

    # test on dev set
    net.eval()
    accurate = 0
    test_nums = 0
    unique_qid_nums = 0
    probs, labels = [], []
    qid_prev = 1
    rank_score = 0.0
    for j, test_data in enumerate(dev_dataloader, 0):
        test_qids, test_questions, test_answers, test_overlap_feats, test_labels = test_data
        test_questions = Variable(test_questions, volatile=True)
        test_answers = Variable(test_answers, volatile=True)
        test_overlap_feats = Variable(test_overlap_feats, volatile=True)
        test_labels = Variable(test_labels.long(), volatile=True)

        if test_qids[0] != qid_prev:
            unique_qid_nums += 1
            probs = torch.Tensor(probs)
            labels = torch.from_numpy(np.array(labels))
            _, accurate_idx = torch.max(labels, 0)
            _, rank_idx = torch.sort(probs, 0, descending=True)
            _, rank = torch.max(rank_idx == accurate_idx[0], 0)
            rank_score += 1/(rank[0]+1)
            probs, labels = [], []
            qid_prev = test_qids[0]

        test_nums += test_questions.size()[0]

        prob = net(test_questions, test_answers, test_overlap_feats)
        # print (prob.data)

        _, predicted = torch.max(prob.data, 1)
        accurate += (predicted == test_labels.data).sum()

        probs.append(5+prob.data[0][1])
        if (j+1)%5000 ==0:
            print(prob.data[0][1])
            MRR_file.writelines(prob_list)
            prob_list = []
        prob_list.append(str(5+prob.data[0][1])+os.linesep)
        labels.append(test_labels.data[0])

        #_, predicted = torch.max(prob.data, 1)
        #right += (predicted == test_labels.data).sum()

        #_, prediction = torch.max(prob.data[:, 1], 0)
        #_, accurate_idx = torch.max(test_labels.data, 0)
        #accurate += (prediction == accurate_idx)[0]
        #_, rank_idx = torch.sort(prob.data[:, 1], 0, descending=True)
        #_, rank = torch.max(rank_idx == accurate_idx[0], 0)
        #rank_score += 1/(rank[0]+1)
        #if (j + 1) == test_steps:
        #    break
    #logger.info('[%d, %5d] test_accuracy: %.3f, MAP: %.3f, MRR: %.3f' % (epoch+1, i+1, right / (test_nums), accurate / test_steps, rank_score / test_steps))

    unique_qid_nums += 1
    probs = torch.Tensor(probs)
    labels = torch.from_numpy(np.array(labels))
    _, accurate_idx = torch.max(labels, 0)
    _, rank_idx = torch.sort(probs, 0, descending=True)
    _, rank = torch.max(rank_idx == accurate_idx[0], 0)
    rank_score += 1/(rank[0]+1)


    print(len(prob_list))
    MRR_file.writelines(prob_list)
    logger.info("Finshed writing MRR")
    logger.info('[%d] test_accuracy: %.3f, MRR: %.3f' % (epoch+1, accurate / test_nums, rank_score / unique_qid_nums))
    log_value('test_accuracy', accurate / test_nums)
    #log_value('MAP', accurate / test_steps)
    log_value('MRR', rank_score / unique_qid_nums)

    logger.info("Finished training")

if __name__ == '__main__':
    train()
