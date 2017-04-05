#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple, defaultdict
from bleu_new import bleu_stats, bleu, smoothed_bleu, bleu_stats_4ref
from numpy.random import choice as nc
import numpy as numpy
import math
import gzip


candidate = namedtuple("candidate", "bleu, features")
epoch = 5
alpha = 10
tau = -50

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



def rank(lst):
    rnk = xrange(0, len(lst))
    return [x+1 for x in sorted(rnk, key=lst.__getitem__, reverse=True)]


def dis(x, y):
    return math.fabs(x - y)


def g(p, q):
    return 1/float(p) - 1/float(q)


def train(cand, w=None, fr_file=None, en_file=None):
    opts = parse_opts(sys.argv)
    # Read reference file to a list.
    ref = en_file
    nbests = [[] for _ in range(len(ref))]

    for k, v in cand.iteritems():
        for x in v:
            i = int(k)
            features = x.features
            stats = smoothed_bleu(list(bleu_stats_4ref(x.translation.split(), ref[i])))
            nbests[i].append(candidate(stats, features))

    sys.stderr.write("Training...\n")
    num_feats = len(nbests[0][0].features)
    weights = numpy.array([0.0] * num_feats) # Initialize weights to all 0
    for e in range(epoch):
        for nbest in nbests:
            f = [numpy.dot(weights, numpy.array(c.features)) for c in nbest]
            y = rank(f)
            n = len(nbest)
            u = numpy.array([0.0] * n)
            for l in range(n):
                for j in range(l):
                    if (y[j] < y[l]) and dis(y[j], y[l]) > alpha and ((f[j] - f[l]) < g(y[j], y[l]) * opts.tau):
                        u[j] = u[j] + g(y[j], y[l])
                        u[l] = u[l] - g(y[j], y[l])
                    if (y[j] > y[l]) and dis(y[j], y[l]) > alpha and ((f[l] - f[j]) < g(y[l], y[j]) * opts.tau):
                        u[j] = u[j] - g(y[l], y[j])
                        u[l] = u[l] + g(y[l], y[j])
            update = sum([u[i] * numpy.array(c.features) for i, c in enumerate(nbest)])
            weights = weights + update
    return weights

