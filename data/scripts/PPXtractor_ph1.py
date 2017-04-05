## This program extracts a phrase-pairs from the word alignments of a parallel corpus ##

import optparse, sys, os, logging, time 
from types import *
from zgc import zgc
from itertools import izip

# Constants
weight_rules = False          # When distributing the unit-count among the rules, should it be weighted by the # of rule occurrences

# Global Variables
rule_indx = 1

srcWrds = []
tgtWrds = []
srcSentlen = 0
tgtSentlen = 0
ruleDict = {}                 # dictionary of rules for each sentence, ruleDict[(src, tgt)] = estimated rule count (1.0 for initial phrase pairs at the begining)
alignDoD = {}                 # Dict of dict to store fwd alignments
revAlignDoD = {}              # Dict of dict to store rev alignments

tgtCntDict = {}
ruleIndxCntDict = {}
fAlignDoD = {}
rAlignDoD = {}

def readSentAlign():
    'Reads the input phrase span file for src & tgt sentences, alignment and initial phrases'

    global opts,ruleDoD
    global ruleDict, phrPairLst
    global srcWrds, tgtWrds, srcSentlen, tgtSentlen

    file_indx = opts.file_prefix
    dDir = opts.datadir
    oDir = opts.outdir
            
    if not dDir.endswith("/"): dDir += "/"
    if not oDir.endswith("/"): oDir += "/"

    srcFile = dDir + file_indx + '.' + opts.src
    tgtFile = dDir + file_indx + '.' + opts.tgt
    alignFile = dDir + file_indx + '.' + opts.alg
    outFile  = oDir + file_indx + '.out'
    outTgtFile  = oDir + 'tgt.' + file_indx + '.out'

    sent_count = 0
    phrLst = []
    aTupLst = []
    totalTime = 0

    print "Using the maximum phrase length           :", opts.max_phr_len
    print "Enforcing tight phrase-pairs constraint   :", opts.tight_phrases_only
    print "Reading the src file                      :", srcFile
    print "Reading the tgt file                      :", tgtFile
    print "Reading the alignment file                :", alignFile
    srcF = open(srcFile, 'r')
    tgtF = open(tgtFile, 'r')
    alignF = open(alignFile, 'r')
    for src_sent, tgt_sent, alignLine in izip(srcF, tgtF, alignF):
        align = alignLine.strip()
        srcWrds = src_sent.strip().split()
        srcSentlen = len(srcWrds)
        tgtWrds = tgt_sent.strip().split()
        tgtSentlen = len(tgtWrds)
        for align_pos in align.split():
            m = align_pos.split('-')
            e = -1 if m[0] == 'Z' else int(m[0])
            f = -1 if m[1] == 'Z' else int(m[1])
            aTupLst.append((e, f))
            try:                                  # Store forward alignments
                alignDoD[m[0]][m[1]] = 1
            except KeyError:
                alignDoD[m[0]] = {}
                alignDoD[m[0]][m[1]] = 1
            try:                                  # Store reverse alignments
                revAlignDoD[m[1]][m[0]] = 1
            except KeyError:
                revAlignDoD[m[1]] = {}
                revAlignDoD[m[1]][m[0]] = 1

        align_tree = zgc(opts.max_phr_len)
        phrPairLst = align_tree.getAlignTree(srcSentlen, tgtSentlen, aTupLst)
        if not opts.tight_phrases_only: phrPairLst = addLoosePhrases(phrPairLst)
        if opts.max_phr_len >= srcSentlen and not ((0, srcSentlen-1),(0, tgtSentlen-1)) in phrPairLst:
            phrPairLst.append(((0, srcSentlen-1),(0, tgtSentlen-1)))
        for ppair in phrPairLst:
            unaligned_edge = False
            # If the boundary term of source or target phrase has an unaligned word, ignore the phrase-pair
            # Earlier bug fixed on March '09
            # Unless the tight-phrase options is set to False
            if not alignDoD.has_key( str(ppair[0][0]) ) or not revAlignDoD.has_key( str(ppair[1][0]) ) or \
               not alignDoD.has_key( str(ppair[0][1]) ) or not revAlignDoD.has_key( str(ppair[1][1]) ):
                if opts.tight_phrases_only: continue
                
            if abs(ppair[1][0] - ppair[1][1]) >= opts.max_phr_len: continue
            if abs(ppair[0][0] - ppair[0][1]) >= opts.max_phr_len: continue
            init_phr_pair = (' '.join( [str(x) for x in xrange(ppair[0][0], ppair[0][1]+1) ] ), \
               ' '.join( [str(x) for x in xrange(ppair[1][0], ppair[1][1]+1)] ) )
            if init_phr_pair in ruleDict: ruleDict[init_phr_pair] += 1.0
            else: ruleDict[init_phr_pair] = 1.0
                
            # Create a dict of dict for storing initial phrase pairs (tuples of source and target spans)
            
        # For every extracted phrase pair call the function compFeatureCounts() to:
        #   i. convert the word positions in the phrase pair into lexical entries, and
        #   ii. find the alignment for the phrase pair and compute the joint count p(s, t)
        for rule in ruleDict.keys(): compFeatureCounts(rule)            
            
        # Clear the variables at the end of current sentence
        alignDoD.clear()
        revAlignDoD.clear()
        ruleDict.clear()
        del aTupLst[:]
        sent_count += 1
        if sent_count % 2000 == 0:
            print "Sentences processed : %6d ..." % sent_count
           

    # Write the rule counts, forward and reverse alignments to files
    with open(outFile, 'w') as oF:
        for rule in sorted( ruleIndxCntDict.iterkeys() ):
            r_indx, rule_count = ruleIndxCntDict[rule]
            f_alignments = ' ## '.join( fAlignDoD[r_indx].keys() )
            r_alignments = ' ## '.join( rAlignDoD[r_indx].keys() )
            oF.write( "%s ||| %g ||| %s ||| %s\n" % (rule, rule_count, r_alignments, f_alignments) )

    with open(outTgtFile, 'w') as tF:
        for tgt in sorted( tgtCntDict.iterkeys() ):
            tF.write( "%s ||| %g\n" % (tgt, tgtCntDict[tgt]) )

    return None

def addLoosePhrases(phr_lst):
    global alignDoD, revAlignDoD, tgtSentlen, srcSentlen
    full_phr_lst = set(phr_lst)
    length=[srcSentlen, tgtSentlen]
    alignDict = [alignDoD, revAlignDoD]
    for tight_ppair in phr_lst:
        curr_lst = set()
        curr_lst.add(tight_ppair)
        for st_ind in [0,1]:
            for p_ind,step in enumerate([-1, 1]):
                new_lst = set()
                for ppair in curr_lst:
                    j = tight_ppair[st_ind][p_ind] + step
                    while j >= 0 and j < length[st_ind]:
                        if alignDict[st_ind].has_key(str(j)):
                            break
                        if p_ind == 0: new_phr = (j, ppair[st_ind][1])
                        elif p_ind == 1: new_phr = (ppair[st_ind][0], j)
                        if st_ind == 0: new_ppair = (new_phr, ppair[1])
                        elif st_ind == 1: new_ppair = (ppair[0], new_phr)
                        new_lst.add(new_ppair)
                        j += step
                curr_lst.update(new_lst)
        full_phr_lst.update(curr_lst)
        
    return list(full_phr_lst)
        
def compFeatureCounts(rule):
    'Convert to lexical phrase and find the alignment for the entries in the phrase. Also compute feature counts P(s|t), P(t|s), P_w(s|t) and P_w(t|s)'

    global srcWrds, tgtWrds
    global fAlignDoD, rAlignDoD
    srcLexLst = []
    tgtLexLst = []
    alignLst = []

    sPosLst = rule[0].split()
    tPosLst = rule[1].split()
    # Convert the word positions in source side of the rule to corresponding lexemes
    item_indx = 0
    for s_tok in sPosLst:
        if s_tok.startswith('X__'):
            srcLexLst.append(s_tok)
        else:
            srcLexLst.append(srcWrds[int(s_tok)])
            # Find the forward alignment for the lexemes in the rule
            alignment = getFwrdAlignment(item_indx, s_tok, tPosLst)
            alignLst.append(alignment)
            #if len(alignment) > 0:
            #    alignLst.append(alignment)
        item_indx += 1
    fAlignment = ' '.join(alignLst)

    # Convert the word positions in target side of the rule to corresponding lexemes
    del alignLst[:]
    item_indx = 0
    for t_tok in tPosLst:
        if t_tok.startswith('X__'):
            tgtLexLst.append(t_tok)
        else:
            tgtLexLst.append(tgtWrds[int(t_tok)])
            # Find the reverse alignment for the lexemes in the rule
            alignment = getRvrsAlignment(item_indx, t_tok, sPosLst)
            alignLst.append(alignment)
            #if len(alignment) > 0:
            #    alignLst.append(alignment)
        item_indx += 1
    rAlignment = ' '.join(alignLst)

    # Get the lexical rule and add its count from the current sentence to total count so far
    curr_rindx = updateRuleCount(' '.join(srcLexLst), ' '.join(tgtLexLst), rule)

    # Update forward and reverse alignment dicts
    f_align_indx = getAlignIndex(fAlignment)
    r_align_indx = getAlignIndex(rAlignment)
    if not fAlignDoD.has_key(curr_rindx):
        fAlignDoD[curr_rindx] = {}
        rAlignDoD[curr_rindx] = {}
    if not fAlignDoD[curr_rindx].has_key(f_align_indx):
        fAlignDoD[curr_rindx][f_align_indx] = 1
    if not rAlignDoD[curr_rindx].has_key(r_align_indx):
        rAlignDoD[curr_rindx][r_align_indx] = 1


def updateRuleCount(mc_src, mc_tgt, rule):
    ''' Updates phrase and target counts '''

    global rule_indx, ruleDict, ruleIndxCntDict, tgtCntDict
    if not tgtCntDict.has_key(mc_tgt):
        tgtCntDict[mc_tgt] = 0
    tgtCntDict[mc_tgt] += ruleDict[rule]

    mc_key = mc_src + ' ||| ' + mc_tgt              # ' ||| ' is the delimiter separating items in the key/value
    if ruleIndxCntDict.has_key(mc_key):
        curr_rindx, curr_cnt = ruleIndxCntDict[mc_key]
        ruleIndxCntDict[mc_key] = ( curr_rindx, curr_cnt + ruleDict[rule] )
    else:
        ruleIndxCntDict[mc_key] = (rule_indx, ruleDict[rule])
        curr_rindx = rule_indx
        rule_indx += 1
    return curr_rindx


def getAlignIndex(align_str):

    tmpLst = align_str.split(' ')
    tmpLst.sort()
    aindx = ''.join(tmpLst)
    return aindx.replace('-', '')


def getFwrdAlignment(item_indx, s_pos, tPosLst):
    'Computes the alignment and lexical weights in forward direction'

    alignLst = []
    if alignDoD.has_key(s_pos):
        alignKeyLst = alignDoD[s_pos].keys()
        alignKeyLst.sort()
        for t_pos in alignKeyLst:
            try:
                # Get the alignment and append it to the list
                alignment = str(item_indx) + '-' + str(tPosLst.index(t_pos))
                alignLst.append(alignment)
            except ValueError:
                pass
    else:
        alignLst.append( str(item_indx) + '-Z' )     # 'Z' represents 'NULL' (i.e. word is unaligned)

    return ' '.join(alignLst)


def getRvrsAlignment(item_indx, t_pos, sPosLst):
    'Computes the alignment and lexical weights in reverse direction'

    alignLst = []
    if revAlignDoD.has_key(t_pos):
        alignKeyLst = revAlignDoD[t_pos].keys()
        alignKeyLst.sort()
        for s_pos in alignKeyLst:
            try:
                # Get the alignment and append it to the list
                alignment = str(sPosLst.index(s_pos)) + '-' + str(item_indx)
                alignLst.append(alignment)
            except ValueError:
                pass
    else:
        alignLst.append( 'Z-' + str(item_indx) )     # 'Z' represents 'NULL' (i.e. word is unaligned)

    return ' '.join(alignLst)


if __name__ == '__main__':
    global opts
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir", default="corpora", help="data directory (default=corpora)")
    optparser.add_option("-o", "--outdir", dest="outdir", default="phrase-pairs", help="data directory (default=phrase-pairs)")
    optparser.add_option("-p", "--prefix", dest="file_prefix", default="train", help="prefix of parallel data files (default=train)")
    optparser.add_option("-e", "--target", dest="tgt", default="en", help="suffix of English (target language) filename (default=en)")
    optparser.add_option("-f", "--source", dest="src", default="cn", help="suffix of French (source language) filename (default=cn)")
    optparser.add_option("-a", "--alignment", dest="alg", default="align", help="suffix of alignment filename (default=align)")
    optparser.add_option("-l", "--logfile", dest="log_file", default=None, help="filename for logging output")
    optparser.add_option("","--tightPhrase", dest="tight_phrases_only", default=False, action="store_true", help="extract just tight-phrases (default=False)")
    optparser.add_option("", "--maxPhrLen", dest="max_phr_len", default=7, type="int", help="maximum phrase length (default=7)")
    (opts, _) = optparser.parse_args()

    if opts.log_file:
        logging.basicConfig(filename=opts.log_file, filemode='w', level=logging.INFO)

    xtract_begin = time.time()
    readSentAlign()
    xtrct_time = time.time() - xtract_begin
    sys.stderr.write( "Phrase extraction time :: %1.7g sec\n\n" % (xtrct_time) )
