#!/usr/bin/env python 
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import logging
import os.path
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


    class_sample_count = [1, 1]
    weight_per_class = 1 / torch.Tensor(class_sample_count).double()
    train_data = np.load('./data/full/data.npz')
    # dev_data = np.load('./data/dev/data.npz')
    test_data = np.load('./data/test/data.npz')
    train_dataset = Dataset(train_data)#, train=True)
    # dev_dataset = Dataset(dev_data)#, train=False)
    test_dataset = Dataset(test_data, test=True)
    weights = [weight_per_class[label] for label in train_data['labels']]
    sampler = data.sampler.WeightedRandomSampler(weights, len(weights))
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler)#, batch_size=batch_size, sampler=sampler)
    # dev_dataloader = data.DataLoader(dev_dataset)#, shuffle=True)
    # dataloader = train_dataloader+dev_dataloader

    net = Net()
    criterion = nn.CrossEntropyLoss()
    ignored_params = list(map(id, net.sentence.conv_q.parameters())) + list(map(id, net.sentence.conv_a.parameters()))
    #ignored_params = list(map(id, net.sentence.conv.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = optim.Adam([
        {'params': net.sentence.conv_q.parameters()},
        {'params': net.sentence.conv_a.parameters()},
        #{'params': net.sentence.conv.parameters()},
        {'params': base_params}
        ], lr=0.000003)

    latest_epoch_num = 9
    model_path = './model/epoch_' + str(latest_epoch_num) + '_2017-05-24#20_07_39.params'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        logger.info('Successfully loaded model: %s' % (model_path))
    else:
        logger.info('Could not find model: %s' % (model_path))

    for epoch in range(n_epochs):
        net.train()
        epoch += latest_epoch_num
        running_loss = 0.0
        correct = 0
        for i, train_data in enumerate(train_dataloader, 0):
            train_qids, train_questions, train_answers, train_overlap_feats, train_labels = train_data
            train_questions = Variable(train_questions)
            train_answers = Variable(train_answers)
            train_overlap_feats = Variable(train_overlap_feats)
            train_labels = Variable(train_labels.long())

            prob = net(train_questions, train_answers, train_overlap_feats)
            loss = criterion(prob, train_labels)
            loss.backward()

            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            #optimizer.zero_grad()
            #optimizer.step()

            running_loss += loss.data[0]
            _, predicted = torch.max(prob.data, 1)
            correct += (predicted == train_labels.data).sum()

            if (i + 1) % train_interval == 0:
                logger.info('[%d, %5d] train_loss: %.3f, train_accuracy: %.3f' % (epoch+1, i+1, running_loss / train_interval, correct / train_interval))
                log_value('train_loss', running_loss / train_interval)
                log_value('train_accuracy', correct / train_interval)
                running_loss = 0.0
                correct = 0

        logger.info("Finished %s epoch" % (epoch+1))
        torch.save(net.state_dict(), './model/epoch_%s_%s.params' % (epoch+1, datetime.datetime.now().strftime("%Y-%m-%d#%H:%M:%S")))
        logger.info('Saved model: ./model/epoch_%s_%s.params' % (epoch+1, datetime.datetime.now().strftime("%Y-%m-%d#%H:%M:%S")))

        # test on dev set
        net.eval()
        probs = []
        MRR_file = open('MRR.txt', 'w')
        for j, test_data in enumerate(test_dataset, 0):
            test_qids, test_questions, test_answers, test_overlap_feats = test_data
            test_questions = Variable(test_questions, volatile=True)
            test_answers = Variable(test_answers, volatile=True)
            test_overlap_feats = Variable(test_overlap_feats, volatile=True)

            prob = net(test_questions, test_answers, test_overlap_feats)

            if (j + 1) % 5000 == 0:
                print(prob.data[0][1])
                MRR_file.writelines(probs)
                probs = []
            probs.append(str(5+prob.data[0][1])+os.linesep)

        print(len(probs))
        MRR_file.writelines(probs)
        MRR_file.close()
    logger.info("Finished training")

if __name__ == '__main__':
    train()
