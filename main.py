import optparse
import os
import sys
from decoder.decode import decode
from decoder.models import TM, LM
from reranker.learn_new import train
from time import time


def parse_opts():
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--train_file", dest="train_file",
                         default=os.path.join("data", "dev", "all.cn-en.cn"),
                         help="File with training sentences")
    optparser.add_option("-e", "--english_file", dest="english_file",
                         default=os.path.join("data", "dev", "all.cn-en.en"),
                         help="File with translated training sentences")
    optparser.add_option("-u", "--train_phrase_table", dest="train_phrase_table",
                         default=os.path.join("data", "large", "phrase-table", "dev-filtered", "rules_cnt.final.out"),
                         help="File with phrase table for training")
    optparser.add_option("-v", "--language_model", dest="lm",
                         default=os.path.join("data", "lm", "en.gigaword.3g.filtered.dev_test.arpa.gz"),
                         help="File with training model")
    optparser.add_option("-w", "--test_file", dest="test_file",
                         default=os.path.join("data", "test", "all.cn-en.cn"),
                         help="File with test sentences")
    optparser.add_option("-x", "--test_phrase_table", dest="test_phrase_table",
                         default=os.path.join("data", "large", "phrase-table", "test-filtered", "rules_cnt.final.out"),
                         help="File with phrase table for testing")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=2, type="int",
                         help="Limit on number of translations to consider per phrase (default=10)")
    (opts, _) = optparser.parse_args()
    return opts


def main():
    opts = parse_opts()
    weights = (1.0, 1.0, 1.0, 1.0)

    start_time = time()
    train_tm = TM(opts.train_phrase_table, opts.k)
    print >> sys.stderr, "Gathering training translation model took %d seconds" % int(time() - start_time)

    start_time = time()
    lm = LM(opts.lm)
    print >> sys.stderr, "Gathering language model took %d seconds" % int(time() - start_time)

    for i in range(0, 3):
        print >> sys.stderr, "--Cycle %d -----------------------------" % (i + 1)
        print >> sys.stderr, "Running decoder..."
        start_time = time()
        candidates, foreign_sentences = decode(tm=train_tm,
                                               lm=lm,
                                               weights=weights,
                                               input_filename=opts.train_file,
                                               num_translations=2,
                                               s=2)
        print >> sys.stderr, "Decoding took %d seconds" % int(time()-start_time)
        print >> sys.stderr, "Preparing candidates for reranker..."
        prepared_candidates = dict()
        for ind, s in enumerate(foreign_sentences):
            prepared_candidates[str(ind)] = candidates[s]

        eng_sentences = []
        for i in range(0, 4):
            sentences = []
            for line in open(opts.english_file + str(i)):
                sentences.append(line.strip())
            eng_sentences.append(sentences)
        references = zip(*eng_sentences)
        print references
        print >> sys.stderr, "Running reranker..."
        start_time = time()
        weights = train(nbest=prepared_candidates, fr_file=opts.train_file, references=references)
        print >> sys.stderr, "Reranking took %d seconds" % int(time() - start_time)
        print >> sys.stderr, "Reranking weights: %s" % str(weights)
        print >> sys.stderr, "---------------------------------------"

    print >> sys.stderr, "Setting up test translation model..."
    test_tm = TM(opts.test_phrase_table, opts.k)
    print >> sys.stderr, "Running decoder on test data..."
    start_time = time()
    candidates, foreign_sentences = decode(tm=test_tm,
                                           lm=lm,
                                           weights=weights,
                                           input_filename=opts.test_file,
                                           num_translations=1)
    print >> sys.stderr, "Decoding took %d seconds" % int(time() - start_time)

    with open('output', 'w') as f:
        for s in foreign_sentences:
            f.write('%s\n' % candidates[s][0].translation)
            print candidates[s][0].translation

    exit(0)


if __name__ == '__main__':
    main()
