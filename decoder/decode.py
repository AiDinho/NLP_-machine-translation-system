#!/usr/bin/env python
import optparse
import gzip
import sys
import models
import heapq
from collections import namedtuple, defaultdict
from itertools import groupby
from operator import itemgetter


def parse_opts(args):
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="../data/test/all.cn-en.cn",
                         help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="../data/large/phrase-table/test-filtered/rules_cnt.final.out",
                         help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default="../data/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz",
                         help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=8, type="int", help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=5, type="int", help="Maximum stack size (default=1)")
    optparser.add_option("-d", "--distortion-limit", dest="d", default=5, type="float", help="Limit number of words per phrase")
    optparser.add_option("-z", "--num-translations", dest="num_translations", default=1, type="int", help="Number of translations")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
    opts = optparser.parse_args(args)[0]
    return opts

opts = parse_opts([])


# Some utility functions from score-decoder.py:
def bitmap(sequence):
  """ Generate a coverage bitmap for a sequence of indexes """
  return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)


def bitmap2str(b, n, on='o', off='.'):
    """ Generate a length-n string representation of bitmap b """
    return '' if n==0 else (on if b&1==1 else off) + bitmap2str(b>>1, n-1, on, off)


def getrange(data):
    ranges = []
    for key, group in groupby(enumerate(data), lambda (index, item): index - item):
        group = map(itemgetter(1), group)
        ranges.append(xrange(group[0], group[-1] + 1))
    return ranges


def cand_phrases(ranges, f, d, tm):
    output_phrases = []
    output_idx = []
    for range_i in ranges:
        for j in range_i:
            for k in xrange(j+1, range_i[-1] + 2):
                if f[j:k] in tm and abs(k-j) <= d:
                    output_phrases.append(tuple(f[j:k]))
                    output_idx.append(xrange(j, k))
    return (output_phrases, output_idx)


def decode(tm, lm, weights=(1.0, 1.0, 1.0, 1.0), input_filename=opts.input, num_sents=opts.num_sents,
           num_translations=opts.num_translations, s=opts.s, d=opts.d):
    print >> sys.stderr, "Reading text to translate from %s..." % input_filename
    french = [tuple(line.strip().split()) for line in open(input_filename).readlines()[:num_sents]]

    # tm should translate unknown words as-is with probability 1
    # Adding features to tm that are all 0s
    for word in set(sum(french,())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, tuple([0.0] * len(weights)))]

    sys.stderr.write("Decoding %s...\n" % (input_filename,))
    best_sentences = defaultdict(list)

    count = 1
    for f in french:
        print >> sys.stderr, "Decoding sentence: ", count, '/', len(french), '\r' if count != len(french) else '\n',
        count += 1
        # Generate future cost table
        future_cost_table = dict()
        for phrase_length in range(1, len(f)+1):
            for start in range(0, len(f) - phrase_length +1):
                end = start + phrase_length
                future_cost_table[(start, end)] = 1
                phrases = tm.get(f[start:end], [])
                if phrases:
                    logprob = 0
                    state = tuple()
                    for word in phrases[0].english.split():
                        state, new_logprob = lm.score(state, word)
                        logprob += new_logprob
                    all_features = list(phrases[0].features) + [logprob]
                    score = sum(weight*feature for weight,feature in zip(weights, all_features))
                    future_cost_table[(start, end)] = score
                for mid in range(start+1, end):
                    future_cost_table[(start, end)] = max(future_cost_table[start, mid] + future_cost_table[mid, end],
                                                          future_cost_table[start, end])

        # Define hypothesis tuple and  initial hypothesis
        hypothesis = namedtuple("hypothesis", "features, score, future_cost, lm_state, bit_string, predecessor, phrase")
        initial_hypothesis = hypothesis((0.0, 0.0, 0.0, 0.0), 0.0, future_cost_table[(0, len(f))], lm.begin(), 0, None, None)
        stacks = [{} for _ in f] + [{}]
        stacks[0][lm.begin(), 0] = initial_hypothesis
        coverage_check = bitmap(xrange(len(f)))
        # Iterate over all the stacks. Each iteration will potentially add new hypothesis into other stacks
        for i, stack in enumerate(stacks[:-1]):
            # Iterate over the top 's' number of hypothesis in a stack
            for h in sorted(stack.itervalues(), key=lambda h: -h.score)[:s]: # prune
                cur_str = bitmap2str(coverage_check - h.bit_string, len(f))
                uncovered_indexes = [pos for pos, char in enumerate(cur_str) if char == 'o']
                uncovered_ranges = getrange(uncovered_indexes)
                uncovered_phrases = cand_phrases(uncovered_ranges, f, d, tm)
                for ph, idx in zip(uncovered_phrases[0], uncovered_phrases[1]):
                    j = len(ph) + i
                    bit_string = h.bit_string + bitmap(idx)
                    for phrase in tm[ph]:
                        future_cost = 0.0
                        bit_start = None
                        # Find future cost for phrases not yet decoded
                        for bit_idx, bit_value in enumerate(str(bit_string)):
                            if int(bit_value) == 1:
                                if bit_start:
                                    future_cost += future_cost_table[(bit_start, bit_idx)]
                                bit_start = None
                            else:
                                if not bit_start:
                                    bit_start = bit_idx

                        logprob = 0
                        lm_state = h.lm_state
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = lm.score(lm_state, word)
                            logprob += word_logprob
                        logprob += lm.end(lm_state) if j == len(f) else 0.0
                        features = tuple(sum(x) for x in zip(tuple(list(phrase.features) + [logprob]), h.features))
                        score = sum(weight * feature for weight, feature in zip(weights, features))
                        new_hypothesis = hypothesis(features, score, future_cost, lm_state, bit_string, h, phrase)
                        key = (lm_state, bit_string)
                        # Use new hypothesis if key doesn't exists or if new hypothesis gives a better score
                        if key not in stacks[j] or stacks[j][key].score + stacks[j][key].future_cost < score + future_cost:
                            stacks[j][lm_state, bit_string] = new_hypothesis
        # Getting top amount of guesses from last stack
        winners = heapq.nlargest(num_translations, stacks[-1].itervalues(), key=lambda h: h.score)
        # winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

        def extract_english(h):
            return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

        candidate = namedtuple('candidate', "translation, features")
        for winner in winners:
            best_sentences[f].append(candidate(extract_english(winner), winner.features))
    return best_sentences, french


if __name__ == '__main__':
    if len(sys.argv) > 1:
        global opts
        opts = parse_opts(sys.argv)
    tm = models.TM(opts.tm, opts.k)
    lm = models.LM(opts.lm)
    best_sentences, french = decode(tm=tm, lm=lm)
    for f in french:
        print best_sentences[f][0].translation