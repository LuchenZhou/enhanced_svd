#!/bin/bash

model=$1
task=$2
attn_pattern=$3
sparsity=$4

python -u eval/LongBench/pred_saveload.py \
    --model $model \
    --task $task \
    --method duo_attn \
    --attn_load_dir ${attn_pattern} \
    --sparsity $sparsity \
    --sink_size 64 \
    --recent_size 256 \
    --compress_svd \               
    --svd_rank 1024               