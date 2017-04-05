#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple, defaultdict
from bleu_new import bleu_stats, bleu, smoothed_bleu, bleu_stats_4ref
from numpy.random import choice as nc
import numpy as np


def parse_opts(args):
    optparser = optparse.OptionParser()
    optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("../../reranker", "data", "train.nbest"),
                         help="N-best file")
    optparser.add_option("-t", "--fr", dest="fr", default=os.path.join("../../reranker", "data", "train.fr"),
                         help="french input for training")
    optparser.add_option("-v", "--en", dest="en", default=os.path.join("../../reranker", "data", "train.en"),
                         help="reference english language input")
    (opts, _) = optparser.parse_args()
    return opts


def filter_list(full_list, excludes):
    s = set(excludes)
    return (x for x in full_list if x not in s)


def get_sample(nblist_sentences, tau, alpha):
    # import time
    sampled_list = list()
    i = 0
    while (i <= tau and len(nblist_sentences) >= 2):

        idx = nc(len(nblist_sentences), 2, replace=False)
        choices = [nblist_sentences[n] for n in idx]

        # choose bleu score and perform the rest
        if abs(choices[0][2] - choices[1][2]) > alpha:
            if choices[0][2] > choices[1][2]:
                t = tuple(choices)
                sampled_list.append(t)
                # sampled_list.append(choices[1])
            else:
                sampled_list.append(tuple(reversed(choices)))
                # sampled_list.append(choices[0])
        nblist_sentences = list(filter_list(nblist_sentences, choices))
        i += 1

    return sampled_list


def my_rerank(nblist, w, epoch=5, tau=5000, xi=100, eta=0.1, alpha=0.1):
    mistakes = 0
    for i in xrange(epoch):
        # nbest is of form num:sentence,vector,bleu score
        # for each sentence get sample
        for i, j in nblist.iteritems():

            sampled_list = get_sample(j, tau, alpha)
            # sort sampled list by
            new_list = sorted(sampled_list, key=lambda x: x[0][2] - x[1][2])[:xi + 1]
            # perceptron update
            for i in new_list:
                s1_feature = np.array(i[0][1])
                s2_feature = np.array(i[1][1])
                if sum(w * s1_feature) <= sum(w * s2_feature):
                    mistakes += 1
                    w += eta * (s1_feature - s2_feature)
    return w


def train(nbest, fr_file, references):
    w = np.array([1.0 / (len(nbest['0'][0].features)) for _ in xrange(len(nbest['0'][0].features))])
    fr_sentence = {}

    for index, line in enumerate(open(fr_file)):
        fr_sentence[index] = line.strip()

    for i, j in fr_sentence.iteritems():
        en_list = [(x.translation, x.features, bleu(tuple(bleu_stats_4ref(x.translation, references[i])))) for x in nbest[str(i)]]
        nbest[str(i)] = en_list

    weights = my_rerank(nbest, w, epoch=5, tau=5000, xi=100, eta=0.1, alpha=0.1)
    return weights


if __name__ == "__main__":
    nbest = defaultdict(list)  ## 0,1,2 as key sentence n weights as value in tuple
    opts = parse_opts(sys.argv)
    candidate = namedtuple('candidate', "translation, features")
    for line in open(opts.nbest):
        (i, sentence, features) = line.strip().split("|||")
        features = tuple(float(x) for x in features.strip().split())
        nbest[i.strip()].append(candidate(sentence, features))
    for line in open(opts.nbest):
        (i, sentence, features) = line.strip().split("|||")
        features = [float(h) for h in features.strip().split()]
        w = np.array([1.0 / (len(features)) for _ in xrange(len(features))])
        break
    weights = train(nbest=nbest, fr_file=opts.fr, en_file=opts.en)
    print "\n".join([str(weight) for weight in weights])

