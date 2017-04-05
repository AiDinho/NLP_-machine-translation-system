#!/usr/bin/env python
import optparse, sys, os
import bleu_new

optparser = optparse.OptionParser()
optparser.add_option("-0", "--reference0", dest="reference0", default=os.path.join("data", "test", "all.cn-en.en0"), help="English reference sentences set 0")
optparser.add_option("-1", "--reference1", dest="reference1", default=os.path.join("data", "test", "all.cn-en.en1"), help="English reference sentences set 1")
optparser.add_option("-2", "--reference2", dest="reference2", default=os.path.join("data", "test", "all.cn-en.en2"), help="English reference sentences set 2")
optparser.add_option("-3", "--reference3", dest="reference3", default=os.path.join("data", "test", "all.cn-en.en3"), help="English reference sentences set 3")
(opts,_) = optparser.parse_args()

ref0 = [line.strip().split() for line in open(opts.reference0)]
ref1 = [line.strip().split() for line in open(opts.reference1)]
ref2 = [line.strip().split() for line in open(opts.reference2)]
ref3 = [line.strip().split() for line in open(opts.reference3)]
ref = zip(ref0, ref1, ref2, ref3)
system = [line.strip().split() for line in sys.stdin]

stats = [0 for i in xrange(10)]
for (r,s) in zip(ref, system):
  stats = [sum(scores) for scores in zip(stats, bleu_new.bleu_stats_4ref(s,r))]
print bleu_new.bleu(stats)