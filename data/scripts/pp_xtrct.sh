#!/bin/bash

root=..
email_addr=your-email@sfu.ca
max_phr_len=7
data_prefix=train
split_size=20000
src=cn
tgt=en

script_dir=$root/scripts
data_dir=../large
src_file=$data_dir/$data_prefix.$src
tgt_file=$data_dir/$data_prefix.$tgt
alg_file=$data_dir/$data_prefix.align

split_dir=$data_dir/split-data
mkdir -p $split_dir
phr_dir=$data_dir/phrase-table
mkdir -p $phr_dir

#### split training corpora ####
lines=`wc -l $src_file | cut -d' ' -f1`
nfiles=`echo "$lines / $split_size" | bc`
let tot="$nfiles * $split_size"
let remaining="$lines - $tot"
if [ $remaining -gt 0 ]; then let nfiles="$nfiles + 1"; fi

for ((i=1; i<= $nfiles; i++)); do
    let s="i * $split_size + 1"
    let t="( i + 1 ) * $split_size"
    sed -n "${s},${t}p" $src_file > $split_dir/$i.$src
    sed -n "${s},${t}p" $tgt_file > $split_dir/$i.$tgt
    sed -n "${s},${t}p" $alg_file > $split_dir/$i.align
done

#### xtract phrase pairs ####
log=$data_dir/log					
scripts=$data_dir/scripts				## to keep temporary scripts
mkdir -p $log
mkdir -p $scripts
step1jobids=''
for ((i=1; i<= $nfiles; i++)); do
    echo "python $script_dir/PPXtractor_ph1.py -d $split_dir/ -o $phr_dir/ -p $i" > $scripts/phase1.$i.sh
    jobid=`qsub -l mem=8gb,walltime=2:00:00 -m e -M $email_addr -e $log -o $log $scripts/phase1.$i.sh`
    step1jobids=$step1jobids:$jobid
done

#### filter phrase pairs for dev and test ####
for dataset in dev test 
do
    mkdir -p $phr_dir/$dataset-temp			## tmp directory to save filtered phrase-pairs (1,...,nfiles)
    mkdir -p $phr_dir/$dataset-filtered			## directory to save final filtered phrase-pairs for dev and test
    currset_src=$root/$dataset/all.$src-$tgt.$src	## source side of dev or test data
    jobids=''
    for ((i=1; i<= $nfiles; i++)); do
        echo "python $script_dir/PPXtractor_ph2n3.py $currset_src $i $phr_dir $phr_dir/$dataset-temp $max_phr_len" > $scripts/phase2a.$dataset.$i.sh
    	jobid=`qsub -l mem=8gb,walltime=1:00:00 -m e -M $email_addr -W depend=afterok:${step1jobids:1} -e $log -o $log $scripts/phase2a.$dataset.$i.sh`
        jobids=$jobids:$jobid
    done
    echo "python $script_dir/PPXtractor_ph2.py $phr_dir/$dataset-temp $phr_dir/$dataset-filtered $data_dir/lex.f2e $data_dir/lex.e2f" > $scripts/phase2b.$dataset.sh
    jobid=`qsub -l mem=16gb,walltime=6:00:00 -m e -M $email_addr -W depend=afterok:${jobids:1} -e $log -o $log $scripts/phase2b.$dataset.sh`
    echo "python $script_dir/PPXtractor_ph3.py $phr_dir/$dataset-filtered/rules_cnt_lprob.out rules_cnt.final.out" > $scripts/phase3.$dataset.sh
    jobid=`qsub -l mem=8gb,walltime=2:00:00 -m e -M $email_addr -W depend=afterok:${jobid:1} -e $log -o $log $scripts/phase3.$dataset.sh`
done


