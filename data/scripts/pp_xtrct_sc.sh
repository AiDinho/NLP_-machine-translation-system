#!/bin/bash

root=..
max_phr_len=7
data_prefix=train
dir_name=../toy
src=cn
tgt=en

script_dir=$root/scripts
data_dir=$dir_name

phr_dir=$data_dir/phrase-table
mkdir -p $phr_dir
mkdir -p $phr_dir/temp

python $script_dir/PPXtractor_ph1.py -d $data_dir -o $phr_dir/temp -p $data_prefix --target $tgt --source $src --maxPhrLen $max_phr_len
python $script_dir/PPXtractor_ph2.py $phr_dir/temp $phr_dir $data_dir/lex.f2e $data_dir/lex.e2f
python $script_dir/PPXtractor_ph3.py $phr_dir/rules_cnt_lprob.out phrase_table.out



