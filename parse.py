#!/usr/bin/env python 
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import os.path
import sys
import codecs
from collections import defaultdict

import numpy as np
import gensim

def load_data(fname, test=False):
    lines = codecs.open(fname, 'r', 'utf-8-sig').readlines()
    if not test:
        qids, questions, answers, labels = [], [], [], []
        question_prev = ""
        qid = 0
        for line in lines:
            parts = line.split('\t')
            label, question, answer = int(parts[0]), parts[1], parts[2].strip('\n')
            logger.info(label)
            if question != question_prev:
                qid += 1
            qids.append(qid)
            questions.append(question.split())
            answers.append(answer.split())
            labels.append(label)
            question_prev = question
        return qids, questions, answers, labels
    else:
        qids, questions, answers = [], [], []
        question_prev = ""
        qid = 0
        for line in lines:
            parts = line.split('\t')
            question, answer = parts[0], parts[1].strip('\n')
            if question != question_prev:
                qid += 1
            qids.append(qid)
            questions.append(question.split())
            answers.append(answer.split())
            question_prev = question
        return qids, questions, answers
        
def add_to_vocab(data, vocab):
    for sentence in data:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def word2vec(model_path):
    return gensim.models.Word2Vec.load(model_path)

def embedding(vocab, model):
    embeddings = []
    for w, idx in vocab.iteritems():
        vector = model[w] if w in model else model['<UNKNOWN_TOKEN>']
        embeddings.append(vector)
    embeddings.append([0] * vector.shape[0])
    embeddings = np.array(embeddings).astype('float32')
    return embeddings

def convert2indices(data, vocab):#, dummy_word_idx, max_sent_length):
    data_idx = []
    for sentence in data:
        #ex = np.ones(max_sent_length) * dummy_word_idx
        ex = np.ones(len(sentence))
        for i, word in enumerate(sentence):
            idx = vocab[word]
            ex[i] = idx
        #data_idx.append(ex)
        data_idx.append(ex.astype('int32'))
    #data_idx = np.array(data_idx).astype('int32')
    return data_idx

def compute_dfs(docs):
    word2df = defaultdict(float)
    for doc in docs:
        for w in set(doc):
            word2df[w] += 1.0
    num_docs = len(docs)
    for w, value in word2df.iteritems():
        word2df[w] /= np.math.log(num_docs / value)
    return word2df 

def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
    word2df = word2df if word2df else {}
    stoplist = stoplist if stoplist else set()
    feats_overlap = []
    for question, answer in zip(questions, answers):
        q_set = set([q for q in question if q not in stoplist])
        a_set = set([a for a in answer if a not in stoplist])
        word_overlap = q_set.intersection(a_set)
        overlap = float(len(word_overlap) / (len(q_set) + len(a_set)))
        
        word_overlap = q_set.intersection(a_set)
        df_overlap = 0.0
        for w in word_overlap:
            df_overlap += word2df[w]
        df_overlap /= (len(q_set) + len(a_set))

        feats_overlap.append(np.array([
                                overlap,
                                df_overlap
                            ]))
    return np.array(feats_overlap).astype('float32')

def compute_overlap_idx(questions, answers, stoplist):#, q_max_sent_length, a_max_sent_length):
    stoplist = stoplist if stoplist else []
    q_indices, a_indices = [], []
    for question, answer in zip(questions, answers):
        q_set = set([q for q in question if q not in stoplist])
        a_set = set([a for a in answer if a not in stoplist])
        word_overlap = q_set.intersection(a_set)

        #q_idx = np.ones(q_max_sent_length) * 2
        q_idx = np.zeros(len(question))
        for i, q in enumerate(question):
            #value = 0
            if q in word_overlap:
                #value = 1
                q_idx[i] = 1
            #q_idx[i] = value
        #q_indices.append(q_idx)
        q_indices.append(q_idx.astype('int32'))

        #a_idx = np.ones(a_max_set_length) * 2
        a_idx = np.zeros(len(answer))
        for i, a in enumerate(answer):
            #value = 0
            if a in word_overlap:
                #value = 1
                a_idx[i] = 1
            #a_idx[i] = value
        #a_indices.append(q_idx)
        a_indices.append(a_idx.astype('int32'))
    
    #q_indices = np.vstack(q_indices).astype('int32')
    #a_indices = np.vstack(a_indices).astype('int32')
    return q_indices, a_indices

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    test = False
    # check and process input arguments
    if len(sys.argv) < 5:
        print("Using: python parse.py BoP2017-DBQA.train.seg.txt wiki.zh.txt.model stoplist.txt train (test)")
        sys.exit(1)
    if len(sys.argv) == 5:
        inp, model_path, stoplist, outdir = sys.argv[1:5]
    elif len(sys.argv) == 6:
        inp, model_path, stoplist, outdir, test = sys.argv[1:6]
        if (test != 'test'):
            print("Using: python parse.py BoP2017-DBQA.test.seg.txt wiki.zh.txt.model stoplist.txt train test")
            sys.exit(1)
        test = True

    if not test:
        qids, questions, answers, labels = load_data(inp)
        qids, labels = np.array(qids).astype('int32'), np.array(labels).astype('int32')
    else:
        qids, questions, answers = load_data(inp, test=True)
        qids = np.array(qids).astype('int32')

    q_max_sent_length = max(map(lambda x: len(x), questions))
    a_max_sent_length = max(map(lambda x: len(x), answers))
    print("q_max_sent_length: " + str(q_max_sent_length))
    print("a_max_sent_length: " + str(a_max_sent_length))

    vocab = {}
    vocab = add_to_vocab(questions, vocab)
    vocab = add_to_vocab(answers, vocab)
    #dummy_word_idx = len(vocab)

    model = word2vec(model_path)
    vocab_embeddings = embedding(vocab, model)
    print("vocab_embeddings: " + str(vocab_embeddings.shape))

    questions_idx = convert2indices(questions, vocab)#, dummy_word_idx, q_max_sent_length)
    answers_idx = convert2indices(answers, vocab)#, dummy_word_idx, a_max_sent_length)
    print("questions: " + str(len(questions_idx)))
    print("answers: " + str(len(answers_idx)))

    seen = set()
    unique_questions = []
    for q, qid in zip(questions, qids):
        if qid not in seen:
            seen.add(qid)
            unique_questions.append(q)
    docs = answers + unique_questions
    word2dfs = compute_dfs(docs)

    stoplist = set(open(stoplist).read().split())
    overlap_feats = compute_overlap_features(questions, answers, stoplist=None, word2df=word2dfs)
    overlap_feats_stoplist = compute_overlap_features(questions, answers, stoplist=stoplist, word2df=word2dfs)
    overlap_feats = np.hstack([overlap_feats, overlap_feats_stoplist])
    print("overlap_feats: " + str(overlap_feats.shape))

    q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist)#, q_max_sent_length, a_max_sent_length)
    print("q_overlap_indices: " + str(len(q_overlap_indices)))
    print("a_overlap_indices: " + str(len(a_overlap_indices)))

    np.save(os.path.join(outdir, 'qids.npy'), qids)
    logger.info("Finished Saved " + outdir + "/qids.npy")
    np.save(os.path.join(outdir, 'questions.npy'), questions_idx)
    logger.info("Finished Saved " + outdir + "/questions.npy")
    np.save(os.path.join(outdir, 'answers.npy'), answers_idx)
    logger.info("Finished Saved " + outdir + "/answers.npy")
    if not test:
        np.save(os.path.join(outdir, 'labels.npy'), labels)
        logger.info("Finished Saved " + outdir + "/labels.npy")
    np.save(os.path.join(outdir, 'overlap_feats.npy'), overlap_feats)
    logger.info("Finished Saved " + outdir + "/overlap_feats.npy")
    np.save(os.path.join(outdir, 'q_overlap_indices.npy'), q_overlap_indices)
    logger.info("Finished Saved " + outdir + "/q_overlap_indices.npy")
    np.save(os.path.join(outdir, 'a_overlap_indices.npy'), a_overlap_indices)
    logger.info("Finished Saved " + outdir + "/a_overlap_indices.npy")
    np.save(os.path.join(outdir, 'vocab_embeddings.npy'), vocab_embeddings)
    logger.info("Finished Saved " + outdir + "/vocab_embeddings.npy")

    if not test:
        np.savez(os.path.join(outdir, 'data.npz'), \
                qids=qids, questions=questions_idx, answers=answers_idx, labels=labels, \
                overlap_feats=overlap_feats, q_overlap_indices=q_overlap_indices, a_overlap_indices=a_overlap_indices, \
                vocab_embeddings=vocab_embeddings)
    else:
        np.savez(os.path.join(outdir, 'data.npz'), \
                qids=qids, questions=questions_idx, answers=answers_idx, \
                overlap_feats=overlap_feats, q_overlap_indices=q_overlap_indices, a_overlap_indices=a_overlap_indices, \
                vocab_embeddings=vocab_embeddings)

    logger.info("Finished Saved All.")
